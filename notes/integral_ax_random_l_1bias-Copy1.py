#!/usr/bin/env python
# coding: utf-8

# In[1]:


# PyTorch
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Scheduler - OneCycleLR, CosineAnnealingLR
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

# PyTorch Lightning
import lightning as L

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


# In[2]:


import os
os.environ["WANDB_SILENT"] = "true"

import warnings
warnings.filterwarnings('ignore')


# In[3]:


L.seed_everything(42)


# In[4]:


df_grf = pl.read_parquet("../data/grf_random_l.parquet")
df_grf_int = pl.read_parquet("../data/grf_random_l_int.parquet")


# In[5]:


n_samples = df_grf["group"].n_unique()
n_samples


# In[6]:


print(df_grf, df_grf_int)


# In[7]:


df_grf = df_grf.filter(pl.col("x").is_in([round(x * 0.01, 2) for x in range(101)]))
print(df_grf)


# In[8]:


x = df_grf.filter(pl.col("group") == 0)["x"].to_numpy()
y = df_grf_int.group_by("group", maintain_order=True).agg(pl.col("y"))["y"].explode().to_numpy().reshape(n_samples, -1)
grfs = df_grf.group_by("group", maintain_order=True).agg(pl.col("grf"))["grf"].explode().to_numpy().reshape(n_samples, -1)
grf_ints = df_grf_int.group_by("group", maintain_order=True).agg(pl.col("grf_int"))["grf_int"].explode().to_numpy().reshape(n_samples, -1)

y = y.astype(np.float32)
grfs = grfs.astype(np.float32)
grf_ints = grf_ints.astype(np.float32)

print(f"x: {x.shape}, y: {y.shape}")
print(f"grfs: {grfs.shape}, grf_ints: {grf_ints.shape}")


# ## DeepONet from Scratch

# $$
# G: u \in C[\mathcal{D}] \rightarrow G(u) \in C[\mathcal{R}] \quad \text{where } \mathcal{D}, \mathcal{R} \text{ are compact}
# $$
# $$
# u(x) \overset{G}{\longrightarrow} G(u)(y) = \int_0^y u(x) dx
# $$

# In[9]:


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


# In[10]:


class IntegralData(Dataset):
    def __init__(self, grf, y, grf_int):
        self.grf = torch.tensor(grf)
        self.y = torch.tensor(y)
        self.grf_int = torch.tensor(grf_int)

    def __len__(self):
        return len(self.grf)

    def __getitem__(self, idx):
        return self.grf[idx], self.y[idx], self.grf_int[idx]


# In[11]:


ds_train = IntegralData(grf_train, y_train, grf_int_train)
ds_val = IntegralData(grf_val, y_val, grf_int_val)
ds_test = IntegralData(grf_test, y_test, grf_int_test)


# In[12]:


class DeepONetScratch(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        num_input = hparams["num_input"]
        num_branch = hparams["num_branch"]
        num_output = hparams["num_output"]
        dim_output = hparams["dim_output"]
        hidden_size = hparams["hidden_size"]
        hidden_depth = hparams["hidden_depth"]

        branch_net = [nn.Linear(num_input, hidden_size), nn.GELU()]
        for i in range(hidden_depth-1):
            branch_net.append(nn.Linear(hidden_size, hidden_size))
            branch_net.append(nn.GELU())
        branch_net.append(nn.Linear(hidden_size, num_branch))
        self.branch_net = nn.Sequential(*branch_net)

        trunk_net = [nn.Linear(dim_output, hidden_size), nn.GELU()]
        for _ in range(hidden_depth-1):
            trunk_net.append(nn.Linear(hidden_size, hidden_size))
            trunk_net.append(nn.GELU())
        trunk_net.append(nn.Linear(hidden_size, num_branch))
        self.trunk_net = nn.Sequential(*trunk_net)
        
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, u, y):
        l = y.shape[1]
        branch_out = self.branch_net(u)
        trunk_out = torch.stack([self.trunk_net(y[:, i:i+1]) for i in range(l)], dim=2)
        pred = torch.einsum("bp,bpl->bl", branch_out, trunk_out) + self.bias
        return pred


# In[13]:


def train(model, optimizer, scheduler, train_loader, val_loader, epochs, device):
    model.to(device)

    val_loss = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for u, y, Guy in train_loader:
            u, y, Guy = u.to(device), y.to(device), Guy.to(device)
            optimizer.zero_grad()
            pred = model(u, y)
            loss = F.mse_loss(pred, Guy)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for u, y, Guy in val_loader:
                u, y, Guy = u.to(device), y.to(device), Guy.to(device)
                pred = model(u, y)
                loss = F.mse_loss(pred, Guy)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        scheduler.step()

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch+1})
    return val_loss


# In[14]:


def evaluate(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for u, y, Guy in test_loader:
            u, y, Guy = u.to(device), y.to(device), Guy.to(device)
            pred = model(u, y)
            loss = F.mse_loss(pred, Guy)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    return test_loss


# In[15]:


dl_train = DataLoader(ds_train, batch_size=500, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=500)
dl_test = DataLoader(ds_test, batch_size=500)


# ## Ax for hyperparameter tuning

# In[16]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def evaluate_model(parameters):
    hparams = {
        "num_input": parameters.get("num_input", 100),
        "num_branch": parameters.get("num_branch", 10),
        "num_output": parameters.get("num_output", 100),
        "dim_output": parameters.get("dim_output", 1),
        "hidden_size": parameters.get("hidden_size", 40),
        "hidden_depth": parameters.get("hidden_depth", 3),
        "learning_rate": parameters.get("learning_rate", 1e-2),
        "batch_size": parameters.get("batch_size", 500),
        "epochs": parameters.get("epochs", 200)
    }
    L.seed_everything(42)
    model = DeepONetScratch(hparams)
    
    wandb.init(project="DeepONet-Ax", config=hparams)
    
    optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"])
    scheduler = OneCycleLR(optimizer, max_lr=hparams["learning_rate"], epochs=hparams["epochs"], steps_per_epoch=len(dl_train) // hparams["batch_size"] + 1)
    val_loss = train(model, optimizer, scheduler, dl_train, dl_val, hparams["epochs"], device)
    test_loss = evaluate(model, dl_test, device)
    
    wandb.log({"test_loss": test_loss})
    wandb.finish()

    print(test_loss)
    
    return val_loss * 1e+5


# In[17]:


ax_client = AxClient(verbose_logging=False)


# In[18]:


ax_client.create_experiment(
    name="DeepONet-Tuning",
    parameters=[
        {
            "name": 'num_input',
            "type": 'fixed',
            "value": 100,
        },
        {
            "name": 'num_branch',
            "type": 'choice',
            "values": [10, 20, 30, 40],
            "value_type": "int",
            "is_ordered": True,
            "sort_values": False,
        },
        {
            "name": 'num_output',
            "type": 'fixed',
            "value": 100,
        },
        {
            "name": 'dim_output',
            "type": 'fixed',
            "value": 1,
        },
        {
            "name": 'hidden_size',
            "type": 'choice',
            "values": [40, 80, 120, 160],
            "value_type": "int",
            "is_ordered": True,
            "sort_values": False,
        },
        {
            "name": 'hidden_depth',
            "type": 'choice',
            "values": [2, 3, 4],
            "value_type": "int",
            "is_ordered": True,
            "sort_values": False,
        },
        {
            "name": 'learning_rate',
            "type": 'range',
            "bounds": [1e-4, 1e-2],
            "log_scale": True,
        },
        {
            "name": 'batch_size',
            "type": 'fixed',
            "value": 500,
        },
        {
            "name": 'epochs',
            "type": 'fixed',
            "value": 200,
        },
    ],
    objectives={"evaluate_model": ObjectiveProperties(minimize=True)},
)


# In[19]:


for _ in range(100):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate_model(parameters))


# In[ ]:


best_parameters, values = ax_client.get_best_parameters()
best_parameters


# In[ ]:


ax_client.generation_strategy.trials_as_df


# In[ ]:


render(ax_client.get_optimization_trace())


# In[ ]:


ax_client.get_best_trial()


# In[ ]:


from ax.plot.scatter import interact_fitted, plot_objective_vs_constraints, tile_fitted
from ax.plot.slice import plot_slice


# In[ ]:


ax_model = ax_client.generation_strategy.model


# In[ ]:


render(plot_slice(ax_model, "learning_rate", "evaluate_model"))


# In[ ]:


best_param, _ = ax_client.get_best_parameters()
best_param


# In[ ]:


render(ax_client.get_feature_importances())


# In[ ]:


ax_df = ax_client.get_trials_data_frame()
ax_df


# In[ ]:


# sort dataframe by evaluate_model
ax_df_sorted = ax_df.sort_values("evaluate_model")
ax_df_sorted


# In[ ]:


ax_client.get_best_trial()


# In[ ]:


ax_df[(ax_df["learning_rate"] > 0.0055) & (ax_df["learning_rate"] < 0.0056)]


# In[ ]:


ax_df.iloc[20]["learning_rate"]

