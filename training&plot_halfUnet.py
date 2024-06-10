import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from network.data_loader_forregressor import DatasetforRegressor
from sklearn.metrics import r2_score
from network.halfunet import HalfUNet
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os


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


gpus = 1 if torch.cuda.is_available() else 0

train_folder = "/data2/yyy/LAS-Diffusion/datasets/try_copy/train/"
val_folder = "/data2/yyy/LAS-Diffusion/datasets/try_copy/val/"
test_folder = "/data2/yyy/LAS-Diffusion/datasets/try_copy/test/"
regressor_folder = "./regressor_halfunet/bulk_divided_by_Kb/try_copy/"
batch_size = 64 

model = MyCNN(train_folder=train_folder, val_folder=val_folder, test_folder=test_folder, batch_size=batch_size)
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  
    mode='min', 
    dirpath=regressor_folder + 'ckpts/',
    filename="{epoch}",
    save_top_k=10000,  
    every_n_epochs=1,
    save_last=True,  
    verbose=False
)



# Initialize the PyTorch Lightning Trainer
logger = pl.loggers.TensorBoardLogger(save_dir=regressor_folder + 'logs/')
trainer = pl.Trainer(gpus=gpus, callbacks=[checkpoint_callback], logger=logger)  

# Train the model
if os.path.exists(regressor_folder + 'ckpts/last.ckpt'):
    trainer.fit(model, ckpt_path=regressor_folder + 'ckpts/last.ckpt')
else:
    trainer.fit(model)

# # Test the model
# model = MyCNN.load_from_checkpoint("/data2/yyy/Projects/LAS-Diffusion/regressor_halfunet/bulk_divided_by_Kb/try_copy/ckpts/epoch=156.ckpt").cuda()
# model.eval()  
# trainer.test(model)