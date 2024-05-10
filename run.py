# PyTorch for deep learning
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader
import lightning as L

# Weights & Biases for experiment tracking
import wandb

# Rich for console output
from rich.console import Console
from rich.progress import Progress

# Neural Hamilton modules
from deeponet.model import DeepONet, VAONet, TFONet
from deeponet.data import train_dataset, val_dataset
from deeponet.train import Trainer, VAETrainer

import survey
import os
import json

# Interactive input for defining hyperparameters and model
def define_model():
    model_type = survey.routines.select(
        "Select model type",
        options=["DeepONet", "VAONet", "TFONet"]
    )
    if model_type == 2:
        d_model         = survey.routines.numeric("Enter d_model (e.g. 32)", decimal=False)
        nhead           = survey.routines.numeric("Enter nhead (e.g. 8)", decimal=False)
        dim_feedforward = survey.routines.numeric("Enter dim_feedforward (e.g. 128)", decimal=False)
        num_layers      = survey.routines.numeric("Enter num_layers (e.g. 4)", decimal=False)
        dropout         = survey.routines.numeric("Enter dropout (e.g. 0.1)")
        learning_rate   = survey.routines.numeric("Enter learning_rate (e.g. 1e-2)")
        batch_size      = survey.routines.numeric("Enter batch_size (e.g. 1000)",decimal=False)
        epochs          = survey.routines.numeric("Enter epochs (e.g. 100)", decimal=False)
        power           = survey.routines.numeric("Enter power (e.g. 2.0)")
        hparams = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "power": power
        }
        model = TFONet(hparams)
        run_name = f"tf_{d_model}_{nhead}_{dim_feedforward}_{num_layers}"
    elif model_type == 1:
        hidden_size     = survey.routines.numeric("Enter hidden_size (e.g. 64)", decimal=False)
        num_layers      = survey.routines.numeric("Enter num_layers (e.g. 4)", decimal=False)
        latent_size     = survey.routines.numeric("Enter latent_size (e.g. 10)", decimal=False)
        dropout         = survey.routines.numeric("Enter dropout (e.g. 0.1)")
        learning_rate   = survey.routines.numeric("Enter learning_rate (e.g. 1e-2)")
        kl_weight       = survey.routines.numeric("Enter kl_weight (e.g. 1e-3)")
        batch_size      = survey.routines.numeric("Enter batch_size (e.g. 1000)", decimal=False)
        epochs          = survey.routines.numeric("Enter epochs (e.g. 100)", decimal=False)
        power           = survey.routines.numeric("Enter power (e.g. 2.0)")
        hparams = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "latent_size": latent_size,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "kl_weight": kl_weight,
            "batch_size": batch_size,
            "epochs": epochs,
            "power": power
        }
        model = VAONet(hparams)
        run_name = f"vae_{hidden_size}_{num_layers}_{latent_size}"
    elif model_type == 0:
        num_input           = 100
        hidden_size         = survey.routines.numeric("Enter hidden_size (e.g. 64)", decimal=False)
        num_branch          = survey.routines.numeric("Enter num_branch (e.g. 10)", decimal=False)
        branch_hidden_depth = survey.routines.numeric("Enter branch_hidden_depth (e.g. 4)", decimal=False)
        trunk_hidden_depth  = survey.routines.numeric("Enter trunk_hidden_depth (e.g. 4)", decimal=False)
        num_output          = 100
        dim_output          = 1
        learning_rate       = survey.routines.numeric("Enter learning_rate (e.g. 1e-2)")
        batch_size          = survey.routines.numeric("Enter batch_size (e.g. 1000)", decimal=False)
        epochs              = survey.routines.numeric("Enter epochs (e.g. 500)", decimal=False)
        power               = survey.routines.numeric("Enter power (e.g. 2.0)")
        hparams = {
            "num_input": num_input,
            "num_branch": num_branch,
            "num_output": num_output,
            "dim_output": dim_output,
            "hidden_size": hidden_size,
            "branch_hidden_depth": branch_hidden_depth,
            "trunk_hidden_depth": trunk_hidden_depth,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "power": power
        }
        model = DeepONet(hparams)
        run_name = f"mlp_{hidden_size}_{num_branch}"
    return model, hparams, run_name, model_type

def main():
    L.seed_everything(42)
    progress = Progress()

    # Define model
    model, hparams, run_name, model_type = define_model()

    # Load dataset
    options = ["normal", "more"]
    more_or_much = survey.routines.select("Do you want normal or more?", options=options)
    more_or_much = options[more_or_much]
    ds_train = train_dataset(more_or_much)
    ds_val = val_dataset(more_or_much)
    dl_train = DataLoader(ds_train, batch_size=hparams["batch_size"], shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=hparams["batch_size"], shuffle=False)
    run_name = f"{run_name}_{more_or_much}"

    # Device
    device_count = torch.cuda.device_count()
    if device_count > 1:
        options = [f"cuda:{i}" for i in range(device_count)] + ["cpu"]
        device = survey.routines.select(
            "Select device",
            options=options
        )
        device = options[device]
    elif device_count == 1:
        device = "cuda:0"
    else:
        device = "cpu"
    print(device)
    model.to(device)

    # Optimizer and scheduler
    #optimizer = Adam(model.parameters(), lr=hparams["learning_rate"])
    optimizer = AdamW(model.parameters(), betas=(0.9, 0.98), lr=hparams["learning_rate"])
    scheduler = PolynomialLR(optimizer, total_iters=int(hparams["epochs"]), power=hparams["power"])

    # Trainer
    trainer = VAETrainer(model, optimizer, scheduler, device) if model_type == 1 else Trainer(model, optimizer, scheduler, device)
    wandb.init(project="DeepONet-Integral", config=hparams, name=run_name)
    
    # Train model
    epochs = hparams["epochs"]
    trainer.train(dl_train, progress, epochs=epochs)
    wandb.finish()

    # Save model
    checkpoint_dir = f"checkpoints/{run_name}"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(model.state_dict(), f"{checkpoint_dir}/model.pth")

    # Save hparams
    with open(f"{checkpoint_dir}/hparams.json", "w") as f:
        json.dump(hparams, f)

if __name__ == "__main__":
    main()
