import torch
import torch.nn as nn
from network.model_utils import *
from torch.optim import AdamW,Adam

class HalfUNet(nn.Module):
    def __init__(self,
                 image_size: int = 64,
                 base_channels: int = 1,
                 dim_mults=(1, 2, 4, 8),
                 dropout: float = 0.1,
                 num_heads: int = 1,
                 world_dims: int = 3,
                 attention_resolutions=(4, 8),
                 with_attention: bool = True,
                 verbose: bool = False,
                 ):
        super().__init__()
        self.out_channels = 1
        channels = [base_channels, *
                    map(lambda m: base_channels * m, dim_mults)]
        in_out = list(zip(channels[:-1], channels[1:]))

        self.verbose = verbose
        self.input_emb = conv_nd(world_dims, 1, base_channels, 3, padding=1)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        ds = 1

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            res = image_size // ds
            self.downs.append(nn.ModuleList([
                ResnetBlock_forhalfunet(world_dims, dim_in, dim_out, dropout=dropout),
                nn.Sequential(
                    normalization(dim_out),
                    activation_function(),
                    AttentionBlock(
                        dim_out, num_heads=num_heads)) if ds in attention_resolutions and with_attention else our_Identity(),
                Downsample(
                    dim_out, dims=world_dims) if not is_last else our_Identity()
            ]))
            if not is_last:
                ds *= 2

        mid_dim = channels[-1]
        res = image_size // ds
        self.mid_block1 = ResnetBlock_forhalfunet(
            world_dims, mid_dim, mid_dim, dropout=dropout)
        self.mid_self_attn = nn.Sequential(
            normalization(mid_dim),
            activation_function(),
            AttentionBlock(mid_dim, num_heads=num_heads)
        ) if ds in attention_resolutions and with_attention else our_Identity()
        self.mid_block2 = ResnetBlock_forhalfunet(
            world_dims, mid_dim, mid_dim, dropout=dropout)
        
        self.flatten = nn.Flatten()
        self._feature_size = 8*8*8*8
        self.out = nn.Sequential(
            nn.Linear(self._feature_size, 2048),
            nn.GroupNorm(32, 2048),
            nn.SiLU(),
            nn.Linear(2048, self.out_channels),
        )

    def forward(self, x):
        if self.verbose:
            print("input: ", x.shape)
        x = self.input_emb(x)
        if self.verbose:
            print("input_emb: ", x.shape)
        h = []
        for resnet, self_attn, downsample in self.downs:
            x = resnet(x)
            if self.verbose:
               print("resnet: ", x.shape)
            x = self_attn(x)
            if self.verbose:
               print("self_attn: ", x.shape)
            h.append(x)
            x = downsample(x)
            if self.verbose:
               print("downsample: ", x.shape)
        if self.verbose:
            print("enter bottle neck")
        x = self.mid_block1(x)
        if self.verbose:
            print("mid_block1: ", x.shape)
        x = self.mid_self_attn(x)
        if self.verbose:
            print("mid_self_attn: ", x.shape)
        x = self.mid_block2(x)
        if self.verbose:
            print("mid_block1: ", x.shape)
        x = self.flatten(x)
        if self.verbose:
            print("flatten: ", x.shape)
        x = self.out(x)
        if self.verbose:
            print("out: ", x.shape)
        return x
    
