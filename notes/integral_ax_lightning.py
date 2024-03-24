# %%
# PyTorch
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Scheduler - OneCycleLR, CosineAnnealingLR
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

# PyTorch Lightning
import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.utilities import disable_possible_user_warnings

# wandb
import wandb

# Ax - Hyperparameter Optimization
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
from ax.utils.notebook.plotting import init_notebook_plotting, render

# Split the data into training and test sets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# %%
disable_possible_user_warnings()

import os
os.environ["WANDB_SILENT"]="true"

# %%
L.seed_everything(42)

# %%
df_grf = pl.read_parquet("../data/grf.parquet")
df_grf_int = pl.read_parquet("../data/grf_int.parquet")

# %%
n_samples = df_grf["group"].n_unique()
n_samples

# %%
print(df_grf, df_grf_int)

# %%
df_grf = df_grf.filter(pl.col("x").is_in([round(x * 0.01, 2) for x in range(101)]))
print(df_grf)

# %%
x = df_grf.filter(pl.col("group") == 0)["x"].to_numpy()
y = df_grf_int.group_by("group", maintain_order=True).agg(pl.col("y"))["y"].explode().to_numpy().reshape(n_samples, -1)
grfs = df_grf.group_by("group", maintain_order=True).agg(pl.col("grf"))["grf"].explode().to_numpy().reshape(n_samples, -1)
grf_ints = df_grf_int.group_by("group", maintain_order=True).agg(pl.col("grf_int"))["grf_int"].explode().to_numpy().reshape(n_samples, -1)

y = y.astype(np.float32)
grfs = grfs.astype(np.float32)
grf_ints = grf_ints.astype(np.float32)

print(f"x: {x.shape}, y: {y.shape}")
print(f"grfs: {grfs.shape}, grf_ints: {grf_ints.shape}")

# %% [markdown]
# ## DeepONet from Scratch

# %% [markdown]
# $$
# G: u \in C[\mathcal{D}] \rightarrow G(u) \in C[\mathcal{R}] \quad \text{where } \mathcal{D}, \mathcal{R} \text{ are compact}
# $$
# $$
# u(x) \overset{G}{\longrightarrow} G(u)(y) = \int_0^y u(x) dx
# $$

# %%
n_train = int(0.8 * n_samples)
n_val = int(0.1 * n_samples)
n_test = n_samples - n_train - n_val

grf_train = grfs[:n_train]
grf_val = grfs[n_train:n_train + n_val]
grf_test = grfs[n_train + n_val:]

y_train = y[:n_train]
y_val = y[n_train:n_train + n_val]
y_test = y[n_train + n_val:]

grf_int_train = grf_ints[:n_train]
grf_int_val = grf_ints[n_train:n_train + n_val]
grf_int_test = grf_ints[n_train + n_val:]

# %%
class IntegralData(Dataset):
    def __init__(self, grf, y, grf_int):
        self.grf = torch.tensor(grf)
        self.y = torch.tensor(y)
        self.grf_int = torch.tensor(grf_int)

    def __len__(self):
        return len(self.grf)

    def __getitem__(self, idx):
        return self.grf[idx], self.y[idx], self.grf_int[idx]

# %%
ds_train = IntegralData(grf_train, y_train, grf_int_train)
ds_val = IntegralData(grf_val, y_val, grf_int_val)
ds_test = IntegralData(grf_test, y_test, grf_int_test)

# %%
class DeepONetScratch(L.LightningModule):
    def __init__(self, hparams):
        """
        hparams: dict
        - num_input: int - number of input features (# of sensors)
        - num_branch: int - number of branches (p in the paper)
        - num_output: int - number of output features (# of points in the output)
        - dim_output: int - dimension of the output (here, 1)
        - hidden_size: int - hidden size of the neural networks (width)
        - hidden_depth: int - depth of the neural networks
        - learning_rate: float - learning rate
        - batch_size: int - batch size
        - epochs: int - number of epochs
        """
        super().__init__()
        
        num_input = hparams["num_input"]
        num_branch = hparams["num_branch"]
        num_output = hparams["num_output"]
        dim_output = hparams["dim_output"]
        hidden_size = hparams["hidden_size"]
        hidden_depth = hparams["hidden_depth"]
        learning_rate = hparams["learning_rate"]
        batch_size = hparams["batch_size"]
        epochs = hparams["epochs"]

        branch_net = [nn.Linear(num_input, hidden_size), nn.GELU()]
        for i in range(hidden_depth-1):
            branch_net.append(nn.Linear(hidden_size, hidden_size))
            branch_net.append(nn.GELU())
        branch_net.append(nn.Linear(hidden_size, num_branch))
        self.branch_net = nn.Sequential(*branch_net)

        trunk_net = [nn.Linear(dim_output, hidden_size), nn.GELU()]
        for i in range(hidden_depth-1):
            trunk_net.append(nn.Linear(hidden_size, hidden_size))
            trunk_net.append(nn.GELU())
        trunk_net.append(nn.Linear(hidden_size, num_branch))
        self.trunk_net = nn.Sequential(*trunk_net)
        
        self.bias = nn.Parameter(torch.randn(num_output), requires_grad=True)

        self.num_input = num_input
        self.num_output = num_output
        self.dim_output = dim_output
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.save_hyperparameters()

    def forward(self, u, y):
        """
        Inputs:
        - u: B x m
        - y: B x l (dim = 1)

        Returns:
        - branch_out: B x p
        - trunk_out: B x p x l
        - pred: B x l
        """
        l = y.shape[1]
        branch_out = self.branch_net(u) # B x p
        trunk_out = torch.stack([self.trunk_net(y[:, i:i+1]) for i in range(l)], dim=2) # B x p x l
        pred = torch.einsum("bp,bpl->bl", branch_out, trunk_out) + self.bias
        return pred
    
    def training_step(self, batch, batch_idx):
        # u: B x m, y: B x l, Guy: B x l
        u, y, Guy = batch
        pred = self(u, y)
        loss = F.mse_loss(pred, Guy)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        u, y, Guy = batch
        pred = self(u, y)
        loss = F.mse_loss(pred, Guy)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        u, y, Guy = batch
        pred = self(u, y)
        loss = F.mse_loss(pred, Guy)
        self.log("test_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": OneCycleLR(
                    optimizer, 
                    max_lr=self.learning_rate, 
                    epochs=self.epochs,
                    steps_per_epoch=len(ds_train) // self.batch_size
                ),
                "interval": "step",
                "monitor": "val_loss",
                "strict": True,
            }
        }
    
    # def setup(self, stage):
    #     if stage == "fit":
    #         self.train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device="cuda"), drop_last=True)
    #         self.val_loader = DataLoader(ds_val, batch_size=self.batch_size, drop_last=True)
    #     if stage == "test":
    #         self.test_loader = DataLoader(ds_test, batch_size=self.batch_size, drop_last=True)

    # def train_dataloader(self):
    #     return self.train_loader
    
    # def val_dataloader(self):
    #     return self.val_loader
    
    # def test_dataloader(self):
    #     return self.test_loader

# %%
wandb_logger = WandbLogger(
    project="DeepONet",
)

# %%
dl_train = DataLoader(ds_train, batch_size=256, shuffle=True, generator=torch.Generator(device="cuda"))
dl_val = DataLoader(ds_val, batch_size=256)
dl_test = DataLoader(ds_test, batch_size=256)

# %% [markdown]
# ## Ax for hyperparameter tuning

# %%
def evaluate_model(parameters):
    hparams = {
        "num_input": 100,
        "num_branch": parameters.get("num_branch", 10),
        "num_output": 100,
        "dim_output": 1,
        "hidden_size": parameters.get("hidden_size", 40),
        "hidden_depth": parameters.get("hidden_depth", 3),
        "learning_rate": parameters.get("learning_rate", 1e-2),
        "batch_size": 512,
        "epochs": 200
    }
    model = DeepONetScratch(hparams)
    print(hparams)
    
    trainer = Trainer(
        max_epochs=model.epochs,
        logger=wandb_logger,
        devices=[0],
        accelerator='auto',
        enable_progress_bar=False,
        callbacks=[LearningRateMonitor(logging_interval='epoch')]
    )
    trainer.fit(model, dl_train, dl_val)
    model.eval()
    results = trainer.test(model, dataloaders=dl_test)
    return results[0]["test_loss"]

# %%
ax_client = AxClient(verbose_logging=False)

# %%
ax_client.create_experiment(
    name="DeepONet-Tuning",
    parameters=[
        {
            "name": 'num_branch',
            "type": 'choice',
            "values": [10, 20, 30, 40],
            "value_type": "int"
        },
        {
            "name": 'hidden_size',
            "type": 'choice',
            "values": [20, 40, 60, 80],
            "value_type": "int"
        },
        {
            "name": 'hidden_depth',
            "type": 'choice',
            "values": [2, 3, 4],
            "value_type": "int"
        },
        {
            "name": 'learning_rate',
            "type": 'range',
            "bounds": [1e-4, 2e-2],
            "log_scale": True,
        },
    ],
    objectives={"evaluate_model": ObjectiveProperties(minimize=True)},
)

# %%
for _ in range(10):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate_model(parameters))

# %%
wandb.finish()


