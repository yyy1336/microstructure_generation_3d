import torch
import torch.nn.functional as F
from tqdm import tqdm
from network.model_utils import *
from network.unet import UNetModel
from einops import rearrange, repeat
import numpy as np
from random import random
from functools import partial
from torch import nn
from torch.special import expm1
import sys
import joblib
import pdb
from network.halfunet import HalfUNet
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt

class MyCNN(pl.LightningModule):
    def __init__(self,
                train_folder = "/data2/yyy/LAS-Diffusion/datasets/try_copy/train/",
                val_folder = "/data2/yyy/LAS-Diffusion/datasets/try_copy/val/",
                test_folder = "/data2/yyy/LAS-Diffusion/datasets/try_copy/test/",
                batch_size = 32):
        super(MyCNN, self).__init__()
        self.model = HalfUNet(verbose=False)
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.test_folder = test_folder
        self.batch_size = batch_size

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),lr=0.0001)
        return optimizer

    def training_step(self, batch, batch_idx):
        x = batch['occupancy']
        y = batch['bulk_Kb']
        y_pred = self.model(x)
        y_pred = y_pred.view(-1)
        loss = nn.MSELoss()(y_pred, y)
        self.log("train_loss", loss.clone().detach().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['occupancy']
        y = batch['bulk_Kb']
        y_pred = self.model(x)
        y_pred = y_pred.view(-1)
        loss = nn.MSELoss()(y_pred, y)
        self.log("val_loss", loss.clone().detach().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch['occupancy']
        y = batch['bulk_Kb']
        y_pred = self.model(x)
        y_pred = y_pred.view(-1)
        loss = nn.MSELoss()(y_pred, y)
        self.log("test_loss", loss.clone().detach().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if not hasattr(self, 'y_preds'):
            self.y_preds = []
            self.y_true = []
        self.y_preds.append(y_pred.detach().cpu().numpy())
        self.y_true.append(y.detach().cpu().numpy())
        return loss
    
    def test_epoch_end(self, outputs):
        y_pred = np.concatenate(self.y_preds, axis=0)
        y_true = np.concatenate(self.y_true, axis=0)

        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(y_true)

        plt.rcParams['svg.fonttype'] = 'none'  # 确保 SVG 中的字体保持为文本
        fig, ax = plt.subplots(figsize=(8, 8)) 

        cf = ax.scatter(y_true, y_pred, alpha=0.5, label='$V$', edgecolors='none', facecolor='blue', s=10)
        ax.set_xlabel('True Values', fontsize=20)
        ax.set_ylabel('Predictions', fontsize=20)
        ax.grid(True, linestyle='--', linewidth=1.5,  color=[0.95, 0.95, 0.95], alpha=0.95)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.plot(ax.get_xlim(), ax.get_xlim(), label='Baseline', color='firebrick', ls="-", alpha=0.7)
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        plt.savefig("./regressor_halfunet/bulk_divided_by_Kb/try_copy/tests/Pred_vs_Truth_epoch156.pdf.svg", dpi=400)
        plt.close('all')
        
        r2 = r2_score(y_true, y_pred)
        self.log("R^2", r2, on_epoch=True, prog_bar=True, logger=True)



    def on_train_epoch_end(self):
        self.log("current_epoch", self.current_epoch, logger=True)
        return super().on_train_epoch_end()
    
    def train_dataloader(self):
        _dataset = DatasetforRegressor(dataset_folder=self.train_folder)
        dataloader = DataLoader(_dataset,
                                num_workers=1,
                                batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False)
        self.iterations = len(dataloader)
        print("len(train_dataloader) =", len(dataloader))
        return dataloader

    def val_dataloader(self):
        _dataset = DatasetforRegressor(dataset_folder=self.val_folder)
        dataloader = DataLoader(_dataset,
                                num_workers=1,
                                batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False)
        self.iterations = len(dataloader)
        print("len(val_dataloader) =", len(dataloader))
        return dataloader

    def test_dataloader(self):
        _dataset = DatasetforRegressor(dataset_folder=self.test_folder)
        dataloader = DataLoader(_dataset,
                                num_workers=1,
                                batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False)
        self.iterations = len(dataloader)
        return dataloader



TRUNCATED_TIME = 0.7


class OccupancyDiffusion(nn.Module):
    def __init__(
            self,
            image_size: int = 64,
            base_channels: int = 128,
            attention_resolutions: str = "16,8",
            with_attention: bool = False,
            num_heads: int = 4,
            dropout: float = 0.0,
            verbose: bool = False,
            use_tensor_condition: bool = False,
            eps: float = 1e-6,
            noise_schedule: str = "linear",
    ):
        super().__init__()
        self.image_size = image_size
        if image_size == 8:
            channel_mult = (1, 4, 8)
        elif image_size == 32:
            channel_mult = (1, 2, 4, 8)
        elif image_size == 64:
            channel_mult = (1, 2, 4, 8, 8)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))
        self.eps = eps
        self.verbose = verbose
        self.use_tensor_condition = use_tensor_condition
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')
        self.denoise_fn = UNetModel(
            image_size=image_size,
            base_channels=base_channels,
            dim_mults=channel_mult, dropout=dropout,
            use_tensor_condition=use_tensor_condition,
            world_dims=3,
            num_heads=num_heads,
            attention_resolutions=tuple(attention_ds), with_attention=with_attention,
            verbose=verbose)

    @property
    def device(self):
        return next(self.denoise_fn.parameters()).device

    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times
    def get_sampling_timesteps_uneven(self, batch, device, steps):

        # ## samp2:
        # steps1 = 10 
        # steps2 = 7  
        # steps3 = 5   
        # # 生成从1.0到0.8的张量，间隔为0.02
        # times1 = torch.linspace(1.0, 0.8, steps1 + 1, device=device)
        # # 生成从0.8到0.1的张量，间隔为0.1
        # times2 = torch.linspace(0.8, 0.1, steps2 + 1, device=device)
        # # 生成从0.1到0的张量，间隔为0.02
        # times3 = torch.linspace(0.1, 0.0, steps3 + 1, device=device)
        # # 将两个张量连接起来
        # times = torch.cat((times1[:-1], times2))
        # times = torch.cat((times[:-1], times3))
        # times = repeat(times, 't -> b t', b=batch)
        # times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        # times = times.unbind(dim=-1)
        

        ## samp3:
        steps1 = 10 
        steps2 = 8   
        # 生成从1.0到0.8的张量，间隔为0.02
        times1 = torch.linspace(1.0, 0.8, steps1 + 1, device=device)
        # 生成从0.8到0.1的张量，间隔为0.1
        times2 = torch.linspace(0.8, 0.0, steps2 + 1, device=device)
        # 将两个张量连接起来
        times = torch.cat((times1[:-1], times2))
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)

        ## samp4:
        # steps1 = 15 
        # steps2 = 7   
        # # 生成从1.0到0.7的张量，间隔为0.02
        # times1 = torch.linspace(1.0, 0.7, steps1 + 1, device=device)
        # # 生成从0.7到0.1的张量，间隔为0.1
        # times2 = torch.linspace(0.7, 0.0, steps2 + 1, device=device)
        # # 将两个张量连接起来
        # times = torch.cat((times1[:-1], times2))
        # times = repeat(times, 't -> b t', b=batch)
        # times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        # times = times.unbind(dim=-1)
        return times

    def training_loss(self, img, tensor_feature,  *args, **kwargs):
        batch = img.shape[0]

        times = torch.zeros(
            (batch,), device=self.device).float().uniform_(0, 1)
        noise = torch.randn_like(img)

        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_img = alpha * img + sigma * noise
        self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                self_cond = self.denoise_fn(
                    noised_img, noise_level,tensor_feature).detach_()
        pred = self.denoise_fn(noised_img, noise_level, tensor_feature, self_cond)

        return F.mse_loss(pred, img)


    @torch.no_grad()
    def sample_unconditional_interp(self, img1, img2, num_generate, steps=50, truncated_index:float=0.0,  verbose: bool = True):
        image_size = self.image_size
        batch, device = 1, self.device

        time_pairs = self.get_sampling_timesteps(
            batch, device=device, steps=steps)

        tensor_zero = - np.ones((4, ),dtype=np.float32)
        tensor_zero = torch.from_numpy(tensor_zero).to(
            device).unsqueeze(0).repeat(batch, 1).to(torch.float32)


        sigma0 = 0.0
        img = img1 * math.sqrt(1 - sigma0 ** 2) + torch.randn_like(img1) * sigma0

        x_start = None
        reverse_time_pairs = time_pairs[::-1]
        for time_pred, time in reverse_time_pairs[1:]:
            log_snr = self.log_snr(time)
            log_snr_pred = self.log_snr(time_pred)
            log_snr, log_snr_pred = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_pred))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_pred, sigma_pred = log_snr_to_alpha_sigma(log_snr_pred)

            noise_cond = self.log_snr(time)
            x_start = self.denoise_fn(img, noise_cond, tensor_zero, None) 

            if time[0] < TRUNCATED_TIME:
                x_start.sign_()

            x_start.clamp_(-1, 1)
            pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)
            img = x_start * alpha_pred + pred_noise * sigma_pred
        noised_img1 = img

        sigma0 = 0.01
        img = img2 * math.sqrt(1 - sigma0 ** 2) + torch.randn_like(img2) * sigma0

        for time_pred, time in reverse_time_pairs[1:]:
            log_snr = self.log_snr(time)
            log_snr_pred = self.log_snr(time_pred)
            log_snr, log_snr_pred = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_pred))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_pred, sigma_pred = log_snr_to_alpha_sigma(log_snr_pred)

            noise_cond = self.log_snr(time)
            x_start = self.denoise_fn(img, noise_cond, tensor_zero, None)

            if time[0] < TRUNCATED_TIME:
                x_start.sign_()

            x_start.clamp_(-1, 1)
            pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)

            img = x_start * alpha_pred + pred_noise * sigma_pred
        noised_img2 = img

        res_tensor = []
        for i in range(num_generate):
            theta = i / (num_generate - 1) * math.pi / 2
            img = noised_img1 * math.cos(theta) + noised_img2 * math.sin(theta)

            x_start = None

            if verbose:
                _iter = tqdm(time_pairs, desc='sampling loop time step')
            else:
                _iter = time_pairs
            for time, time_next in _iter:

                log_snr = self.log_snr(time)
                log_snr_next = self.log_snr(time_next)
                log_snr, log_snr_next = map(
                    partial(right_pad_dims_to, img), (log_snr, log_snr_next))

                alpha, sigma = log_snr_to_alpha_sigma(log_snr)
                alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

                noise_cond = self.log_snr(time)
                x_start = self.denoise_fn(img, noise_cond, tensor_zero, x_start)

                if time[0] < TRUNCATED_TIME:
                    x_start.sign_()

                # DDIM:
                x_start.clamp_(-1, 1)
                pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)
                img = x_start * alpha_next + pred_noise * sigma_next

                # DDPM:
                # c = -expm1(log_snr - log_snr_next)
                # mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
                # variance = (sigma_next ** 2) * c
                # noise = torch.where(
                #     rearrange(time_next > truncated_index, 'b -> b 1 1 1 1'),
                #     torch.randn_like(img),
                #     torch.zeros_like(img)
                # )
                # img = mean + torch.sqrt(variance) * noise
            res_tensor.append(img)
        return res_tensor


    @torch.no_grad()
    def sample_cls_interp(self, img1, img2, tensors, num_generate,
                             steps=50, truncated_index:float=0.0, verbose: bool = True, cls_model_path: bool = " "):
        clsmodel = (MyCNN.load_from_checkpoint(cls_model_path).cuda()).model
        image_size = self.image_size
        shape = (1, 1, image_size, image_size, image_size)
        batch, device = 1, self.device

        time_pairs = self.get_sampling_timesteps(
            batch, device=device, steps=steps)

        tensor_zero = - np.ones((4, ),dtype=np.float32)
        tensor_zero = torch.from_numpy(tensor_zero).to(
            device).unsqueeze(0).repeat(batch, 1).to(torch.float32)

        tensor1 = tensors[0, :]
        tensor1 = torch.from_numpy(tensor1).to(
            device).unsqueeze(0).repeat(batch, 1).to(torch.float32)
        tensor2 = tensors[-1, :]
        tensor2 = torch.from_numpy(tensor2).to(
            device).unsqueeze(0).repeat(batch, 1).to(torch.float32)
        
        sigma0 = 0.01 
        img = img1 * math.sqrt(1 - sigma0 ** 2) + torch.randn_like(img1) * sigma0

        x_start = None
        reverse_time_pairs = time_pairs[::-1]
        for time_pred, time in reverse_time_pairs[1:]:
            log_snr = self.log_snr(time)
            log_snr_pred = self.log_snr(time_pred)
            log_snr, log_snr_pred = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_pred))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_pred, sigma_pred = log_snr_to_alpha_sigma(log_snr_pred)

            noise_cond = self.log_snr(time)
            x_start = self.denoise_fn(img, noise_cond, tensor_zero, x_start) # self-conditioning

            if time[0] < TRUNCATED_TIME:
                x_start.sign_()

            x_start.clamp_(-1, 1)
            pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)
            img = x_start * alpha_pred + pred_noise * sigma_pred
        noised_img1 = img


        sigma0 = 0.01
        img = img2 * math.sqrt(1 - sigma0 ** 2) + torch.randn_like(img2) * sigma0

        for time_pred, time in reverse_time_pairs[1:]:
            log_snr = self.log_snr(time)
            log_snr_pred = self.log_snr(time_pred)
            log_snr, log_snr_pred = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_pred))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_pred, sigma_pred = log_snr_to_alpha_sigma(log_snr_pred)

            noise_cond = self.log_snr(time)
            x_start = self.denoise_fn(img, noise_cond, tensor_zero, x_start)

            if time[0] < TRUNCATED_TIME:
                x_start.sign_()

            # DDIM reverse sample:
            x_start.clamp_(-1, 1)
            pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)

            img = x_start * alpha_pred + pred_noise * sigma_pred
        noised_img2 = img


        res_tensor = []
        beta = 20
        eta = 0.4
        for i in range(num_generate):
            alpha = i / (num_generate - 1)
            if (i == 0):
                img = noised_img1
            elif (i == num_generate - 1):
                img = noised_img2
            else:
                theta = eta - math.atanh(math.tanh(beta * eta) - i / (num_generate - 1) * (
                            math.tanh(beta * eta) - math.tanh(beta * (eta - 1)))) / beta
                theta = theta * math.pi / 2
                # theta = i / (num_generate - 1) * math.pi / 2
                img = noised_img1 * math.cos(theta) + noised_img2 * math.sin(theta)

            

            vol = torch.from_numpy(tensors[i,:]).to(device)
            tensor = tensor_zero
            tensor[0,3]=vol
            
            x_start = None
            if verbose:
                _iter = tqdm(time_pairs, desc='sampling loop time step')
            else:
                _iter = time_pairs
            
            for time, time_next in _iter:
                log_snr = self.log_snr(time)
                log_snr_next = self.log_snr(time_next)
                log_snr, log_snr_next = map(
                    partial(right_pad_dims_to, img), (log_snr, log_snr_next))
            
                alpha, sigma = log_snr_to_alpha_sigma(log_snr)
                alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)
            
                noise_cond = self.log_snr(time)
                x_start = self.denoise_fn(img, noise_cond, tensor, x_start)
        
                with torch.enable_grad():
                    img.requires_grad_(True)
                    x_start_copy = None
                    if (x_start is not None):
                        x_start_copy = x_start.detach() 
                    x_start = self.denoise_fn(img, noise_cond, tensor, x_start_copy)

                
                    theta = 64
                    x01 = (torch.tanh(theta * x_start) + 1) / 2
                    obj = - (clsmodel(x01))**2
                    if(i>0):
                        img_last=res_tensor[-1]
                        img_end = torch.ones([1,1,64,64,64]).to(device)
                        img_begin = res_tensor[0]
                        # obj = obj + torch.norm((torch.tanh(theta * x_start[:, :, 0, :, :]) + 1) / 2 - img_last[:, :, 0, :, :], p=2)/(64*64) 
                        obj = 3*obj + torch.norm((torch.tanh(theta * x_start[:, :, 0, :, :]) + 1) / 2 - img_last[:, :, 0, :, :], p=2)/(64*64) + (1-i/(num_generate-1)) * torch.norm((torch.tanh(theta * x_start[:, :, 0, :, :]) + 1) / 2 - img_end[:, :, 0, :, :], p=2)/(64*64) + (i/(num_generate-1)) * torch.norm((torch.tanh(theta * x_start[:, :, 0, :, :]) + 1) / 2 - img_begin[:, :, 0, :, :], p=2)/(64*64)
                
                    obj.backward()
                    gradient_img = img.grad
                
            
                if time[0] < TRUNCATED_TIME:
                    x_start.sign_()
            
                # DDIM:
                x_start.clamp_(-1, 1)
                pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)
                scale = 10
                if (i>0 and i <num_generate-1):
                    pred_noise = pred_noise - scale * sigma_next * gradient_img
            
                img = x_start * alpha_next + pred_noise * sigma_next

            print("(bulk/Kb)_pred = ", clsmodel(img))
            x01 = (torch.tanh(64 * x_start) + 1) / 2
            print("vol = ", x01.sum() / (64 ** 3))
            res_tensor.append(img)
        return res_tensor

    @torch.no_grad()
    def sample_unconditional(self, batch_size=16,
                             steps=50, truncated_index:float=0.0, verbose: bool = True):
        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size, image_size)
        
        batch, device = shape[0], self.device

        time_pairs = self.get_sampling_timesteps_uneven(batch, device=device, steps=steps)

        tensor_zero = - np.ones((10,), dtype=np.float32)
        tensor_zero = torch.from_numpy(tensor_zero).to(
            device).unsqueeze(0).repeat(batch, 1).to(torch.float32)

        img = torch.randn(shape, device=device)
        x_start = None

        if verbose:
            _iter = tqdm(time_pairs, desc='sampling loop time step')
        else:
            _iter = time_pairs

        for time, time_next in _iter:
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time)
            x_start = self.denoise_fn(
                img, noise_cond, tensor_zero, x_start)

            if time[0] < TRUNCATED_TIME:
                x_start.sign_()
            
            # DDIM:
            x_start.clamp_(-1, 1)
            pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)
            img = x_start * alpha_next + pred_noise * sigma_next
        
            
            #DDPM:
            # c = -expm1(log_snr - log_snr_next)
            # mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            # variance = (sigma_next ** 2) * c
            # noise = torch.where(
            #     rearrange(time_next > truncated_index, 'b -> b 1 1 1 1'),
            #     torch.randn_like(img),
            #     torch.zeros_like(img)
            # )
            # img = mean + torch.sqrt(variance) * noise

        return img

    @torch.no_grad()
    def sample_with_tensor(self, tensor_c, batch_size=16,
                           steps=50,  truncated_index:float=0.0, tensor_w: float = 1.0, verbose: bool = True):
        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size, image_size)
        batch, device = shape[0], self.device

        time_pairs = self.get_sampling_timesteps(batch, device=device, steps=steps)

        tensor_condition = torch.from_numpy(tensor_c).to(
            device).unsqueeze(0).repeat(batch, 1).to(torch.float32)
        tensor_zero = - np.ones((4,), dtype=np.float32)  
        tensor_zero = torch.from_numpy(tensor_zero).to(
            device).unsqueeze(0).repeat(batch, 1).to(torch.float32)

        img = torch.randn(shape, device=device)
        x_start = None

        if verbose:
            _iter = tqdm(time_pairs, desc='sampling loop time step')
        else:
            _iter = time_pairs
        for time, time_next in _iter:
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time)
            x_zero_none = self.denoise_fn(
                img, noise_cond, tensor_zero, x_start)

          
            x_start = x_zero_none + tensor_w * \
                        (self.denoise_fn(img, noise_cond, tensor_condition, x_start) - x_zero_none)

            if time[0] < TRUNCATED_TIME:
                x_start.sign_()

            # DDIM:
            x_start.clamp_(-1, 1)
            pred_noise = (img - alpha * x_start) / \
                            sigma.clamp(min=1e-8)

            img = x_start * alpha_next + pred_noise * sigma_next
        return img
