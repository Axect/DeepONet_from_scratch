import torch

from deeponet.model import DeepONet, VAONet, TFONet, KANON
from deeponet.data import val_dataset
from deeponet.utils import VAEPredictor, Predictor

import numpy as np

import os
import survey
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Choose checkpoint
checkpoints = os.listdir("checkpoints")
chosen = survey.routines.select("Select checkpoint", options=checkpoints)
checkpoint = f"checkpoints/{checkpoints[chosen]}"

# Load the hyperparameters
hparams = json.load(open(f"{checkpoint}/hparams.json", "r"))

# Load the model
# Check if checkpoint contains 'tf' or 'vae' or 'kan'
if "tf" in checkpoint:
    model = TFONet(hparams)
    model.load_state_dict(torch.load(
        f"{checkpoint}/model.pth", map_location=device))
    predictor = Predictor(
        model,
        device=device,
        study_name="DeepONet-Integral",
        run_name=checkpoints[chosen]
    )
elif "vae" in checkpoint:
    model = VAONet(hparams)
    model.load_state_dict(torch.load(
        f"{checkpoint}/model.pth", map_location=device))
    predictor = VAEPredictor(
        model,
        device=device,
        study_name="DeepONet-Integral",
        run_name=checkpoints[chosen]
    )
elif "kan" in checkpoint:
    model = KANON(hparams)
    model.load_state_dict(torch.load(
        f"{checkpoint}/model.pth", map_location=device))
    predictor = Predictor(
        model,
        device=device,
        study_name="DeepONet-Integral",
        run_name=checkpoints[chosen]
    )
else:
    model = DeepONet(hparams)
    model.load_state_dict(torch.load(
        f"{checkpoint}/model.pth", map_location=device))
    predictor = Predictor(
        model,
        device=device,
        study_name="DeepONet-Integral",
        run_name=checkpoints[chosen]
    )

# ==============================================================================
# Validation dataset
# ==============================================================================
# Load the data
ds_val = val_dataset("normal")

for i in range(5):
    u, y, Guy = ds_val[i]
    x = np.linspace(0, 1, len(u))

    # Plot the potential
    predictor.potential_plot(x, u, name=f"grf_val_{i}")

    # Plot the prediction
    predictor.predict_plot(u, y, Guy, name=f"prediction_val_{i}")

# ==============================================================================
# Custom test
# ==============================================================================
# Polynomial
x = torch.linspace(0, 1, 100)
u = 4 * x * (x - 1)  # 4x^2 - 4x
y = torch.linspace(0, 1, 100)
Guy = 4/3 * x**3 - 2 * x**2
predictor.potential_plot(x, u, name=f"poly")
predictor.predict_plot(u, y, Guy, name=f"predicion_poly")

# Exponential
u = torch.exp(x) / np.exp(1)
Guy = (torch.exp(x) - 1) / np.exp(1)
predictor.potential_plot(x, u, name=f"exp")
predictor.predict_plot(u, y, Guy, name=f"predicion_exp")

# Cosine
u = torch.cos(x * 2 * np.pi)
Guy = torch.sin(x * 2 * np.pi) / (2 * np.pi)
predictor.potential_plot(x, u, name=f"cos")
predictor.predict_plot(u, y, Guy, name=f"predicion_cos")
