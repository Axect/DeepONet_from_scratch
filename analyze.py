# Optuna for hyperparameter tuning
import optuna

# PyTorch for deep learning
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, PolynomialLR
from torch.utils.data import DataLoader

# Weights & Biases for experiment tracking
import wandb

# Plotly for visualization
from plotly.offline import plot
import plotly.graph_objects as go

# Rich for console output
from rich.console import Console
from rich.progress import Progress

# DeepONet modules
from deeponet.model import DeepONetScratch
from deeponet.data import load_data, train_val_test_split, IntegralData, collate_fn
from deeponet.train import train_epoch, evaluate
from deeponet.utils import create_activation, predict, predict_plot

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Load the best study
study = optuna.load_study(study_name="DeepONet_fix", storage="sqlite:///optuna.db")
best_trial = study.best_trial
checkpoint = best_trial.user_attrs["checkpoint"]
print(f"Best trial: {best_trial.number}")
print(f"Best value: {best_trial.value}")
print(f"Best parameters: {best_trial.params}")
print(f"Checkpoint: {checkpoint}")

# Load the best hyperparameters
hparams = best_trial.params
hparams["num_input"] = 100
hparams["num_output"] = 100
hparams["dim_output"] = 1
hparams["batch_size"] = 500
hparams["epochs"] = 200
# hparams["hidden_activation"] = create_activation(hparams["hidden_activation"])
hparams["hidden_activation"] = create_activation("GELU")

# Load the best model
model = DeepONetScratch(hparams)
model.load_state_dict(torch.load(checkpoint))

# Load data
data_path = "data/"
x, y, grfs, grf_ints, n_samples = load_data(data_path, random=True)
ds = IntegralData(grfs, y, grf_ints)

# Evaluate the best model
model.eval()
(u_test, y_test, Guy_test) = ds[0]

# sort y_test, Guy test
y_test, indices = torch.sort(y_test)
Guy_test = Guy_test[indices]

predict_plot(model, u_test, y_test, Guy_test, "figs/test.png")

# Test for sine function
x = torch.linspace(0, 1, 100).view( -1, 1).to("cuda:0")
u = (np.pi * 2 * x).cos()
y = torch.linspace(0, 1, 100).view( -1, 1).to("cuda:0")
Guy = (np.pi * 2 * y).sin() / (np.pi * 2)

predict_plot(model, u, y, Guy, "figs/test_sin.png")
