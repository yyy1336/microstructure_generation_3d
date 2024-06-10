import torch
from torch import nn
import math
import torch.nn.functional as F
from einops import rearrange


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def activation_function():
    return nn.SiLU()


class our_Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def mask_kernel(x, sigma=1):
    return torch.abs(x) > sigma - 1e-6


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    _channels = min(channels, 32)
    return GroupNorm32(_channels, channels)


class AttentionBlock(nn.Module):

    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def position_encoding(d_model, length):
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class CrossAttention(nn.Module):
    def __init__(self, feature_dim, tensor_dim: int,
                 num_heads: int = 8, image_size: int = 8, world_dims: int = 3, drop_out: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.image_size = image_size
        self.world_dims = world_dims
        self.q = nn.Sequential(
            normalization(feature_dim),
            activation_function(),
            conv_nd(world_dims, feature_dim, feature_dim, 3, padding=1),
        )
        self.k = nn.Linear(tensor_dim, feature_dim)
        self.v = nn.Linear(tensor_dim, feature_dim)

        self.voxel_pe = position_encoding(
            feature_dim, self.image_size ** world_dims)

        self.attn = torch.nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True, dropout=drop_out)

    def forward(self, x, tensor_feature):
        q = self.q(x).reshape(x.shape[0], self.feature_dim, -1).transpose(1,
                                                                          2) + self.voxel_pe.to(x.device).unsqueeze(0)
        k = self.k(tensor_feature)
        v = self.v(tensor_feature)
        attn, _ = self.attn(q, k, v, attn_mask=None)
        return attn.transpose(1, 2).reshape(x.shape[0], self.feature_dim, *(self.image_size,) * self.world_dims)



class QKVAttention(nn.Module):
    def forward(self, qkv):
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)


class Upsample(nn.Module):
    def __init__(self, channels, use_conv=True, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv=True, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2
        if use_conv:
            self.op = conv_nd(dims, channels, channels,
                              3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


def clamp_pixel(pixels):
    return torch.clip(torch.round(pixels), 0, 223)


class ResnetBlock(nn.Module):
    def __init__(self, world_dims: int, dim_in: int, dim_out: int, emb_dim: int, dropout: float = 0.1, 
                 use_tensor_condition: bool = True):
        super().__init__()
        self.world_dims = world_dims
        self.time_mlp = nn.Sequential(
            activation_function(),
            nn.Linear(emb_dim, dim_out)
        )
        self.use_tensor_condition = use_tensor_condition
        if self.use_tensor_condition:
            self.tensor_mlp0 = nn.Sequential(
                activation_function(),
                nn.Linear(emb_dim, dim_out),
            )
            self.tensor_mlp1 = nn.Sequential(
                activation_function(),
                nn.Linear(emb_dim, dim_out),
            )
            self.tensor_mlp2 = nn.Sequential(
                activation_function(),
                nn.Linear(emb_dim, dim_out),
            )
            self.tensor_mlp3 = nn.Sequential(
                activation_function(),
                nn.Linear(emb_dim, dim_out),
            )

        self.block1 = nn.Sequential(
            normalization(dim_in),
            activation_function(),
            conv_nd(world_dims, dim_in, dim_out, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            normalization(dim_out),
            activation_function(),
            nn.Dropout(dropout),
            zero_module(conv_nd(world_dims, dim_out, dim_out, 3, padding=1)),
        )
        self.res_conv = conv_nd(
            world_dims, dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb, tensor_condition=None):
        h = self.block1(x)
        h += self.time_mlp(time_emb)[(...,) + (None,) * self.world_dims]
        if self.use_tensor_condition:
            h += self.tensor_mlp0(tensor_condition[:,:,0])[(...,) + (None,) * self.world_dims]
            h += self.tensor_mlp1(tensor_condition[:,:,1])[(...,) + (None,) * self.world_dims]
            h += self.tensor_mlp2(tensor_condition[:,:,2])[(...,) + (None,) * self.world_dims]
            h += self.tensor_mlp3(tensor_condition[:,:,3])[(...,) + (None,) * self.world_dims]
        h = self.block2(h)
        return h + self.res_conv(x)

class ResnetBlock_forhalfunet(nn.Module):
    def __init__(self, world_dims: int, dim_in: int, dim_out: int, dropout: float = 0.1):
        super().__init__()
        self.world_dims = world_dims
        self.block1 = nn.Sequential(
            normalization(dim_in),
            activation_function(),
            conv_nd(world_dims, dim_in, dim_out, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            normalization(dim_out),
            activation_function(),
            nn.Dropout(dropout),
            zero_module(conv_nd(world_dims, dim_out, dim_out, 3, padding=1)),
        )
        self.res_conv = conv_nd(
            world_dims, dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1') 
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))



def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def beta_linear_log_snr(t):
    return -torch.log(torch.special.expm1(1e-4 + 10 * (t ** 2)))


def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps=1e-5)


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))
