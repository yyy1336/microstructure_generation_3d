import copy
from utils.utils import set_requires_grad
from torch.utils.data import DataLoader
from network.model_utils import EMA
from network.data_loader import occupancy_field_Dataset
from pathlib import Path
from torch.optim import AdamW,Adam
from utils.utils import update_moving_average
from pytorch_lightning import LightningModule
from network.model import OccupancyDiffusion
import torch.nn as nn
import os
import random

class DiffusionModel(LightningModule):
    def __init__(
        self,
        dataset_folder: str = "",
        results_folder: str = './results',
        image_size: int = 32,
        base_channels: int = 32,
        lr: float = 2e-4,
        batch_size: int = 8,
        attention_resolutions: str = "16,8",
        optimizier: str = "adam",
        with_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0,
        ema_rate: float = 0.999,
        verbose: bool = False,
        save_every_epoch: int = 1,
        training_epoch: int = 100,
        gradient_clip_val: float = 1.0,
        use_tensor_condition: bool = False,
        noise_schedule: str = "linear",
        debug: bool = False,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False
        self.results_folder = Path(results_folder)
        self.model = OccupancyDiffusion(image_size=image_size, base_channels=base_channels,
                                        attention_resolutions=attention_resolutions,
                                        with_attention=with_attention,
                                        dropout=dropout,
                                        use_tensor_condition=use_tensor_condition,
                                        num_heads=num_heads,
                                        noise_schedule=noise_schedule,
                                        verbose=verbose)

        self.batch_size = batch_size
        self.lr = lr
        self.image_size = image_size
        self.dataset_folder = dataset_folder
        self.with_attention = with_attention
        self.save_every_epoch = save_every_epoch
        self.traning_epoch = training_epoch
        self.gradient_clip_val = gradient_clip_val
        self.use_tensor_condition = use_tensor_condition
        self.ema_updater = EMA(ema_rate)
        self.ema_model = copy.deepcopy(self.model)
        self.optimizier = optimizier
        self.reset_parameters()
        set_requires_grad(self.ema_model, False)
        if debug:
            self.num_workers = 1
        else:
            self.num_workers = os.cpu_count()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_EMA(self):
        update_moving_average(self.ema_model, self.model, self.ema_updater)

    def configure_optimizers(self):
        if self.optimizier == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizier == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
        return [optimizer]

    def train_dataloader(self):
        _dataset = occupancy_field_Dataset(dataset_folder=self.dataset_folder,
                                           use_tensor_condition=self.use_tensor_condition,
                                           )
        dataloader = DataLoader(_dataset,
                                num_workers=self.num_workers,
                                batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False)
        self.iterations = len(dataloader)
        return dataloader

    def training_step(self, batch, batch_idx):
        occupancy = batch["occupancy"]

        if self.use_tensor_condition:
            tensor_feature = batch["tensor_feature"]
        else:
            tensor_feature = None

        loss = self.model.training_loss(
            occupancy, tensor_feature).mean()

        self.log("loss", loss.clone().detach().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)  
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_val)
        opt.step()


        self.update_EMA()

    def on_train_epoch_end(self):
        self.log("current_epoch",self.current_epoch, logger=True)
        return super().on_train_epoch_end()

