import argparse
import os

# Optuna for hyperparameter tuning
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, WilcoxonPruner, PercentilePruner, HyperbandPruner
from optuna.storages import RetryFailedTrialCallback
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna_integration import BoTorchSampler

# PyTorch for deep learning
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, PolynomialLR, CosineAnnealingLR
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
from deeponet.utils import create_activation

BATCH_SIZE = 500

def parse_arguments():
    parser = argparse.ArgumentParser(description='DeepONet Hyperparameter Tuning with Optuna')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for Optuna (default: 100)')
    return parser.parse_args()

def objective(trial, wandb_group, console, progress, task_id):
    hparams = {
        "num_input": 100,
        "num_branch": trial.suggest_categorical("num_branch", [10, 20, 30]),
        "num_output": 100,
        "dim_output": 1,
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "hidden_depth": trial.suggest_int("hidden_depth", 2, 4),
        # "hidden_activation": create_activation(trial.suggest_categorical("hidden_activation", ['ReLU', 'GELU', 'SiLU', 'Mish'])),
        "hidden_activation": create_activation("Mish"),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        # "power": trial.suggest_float("power", 0.5, 2.5),
        "power": 2.0,
        "batch_size": BATCH_SIZE,
        "epochs": 200
    }

    model = DeepONetScratch(hparams)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=hparams["learning_rate"])
    # scheduler = OneCycleLR(optimizer, max_lr=hparams["learning_rate"], epochs=hparams["epochs"], steps_per_epoch=len(dl_train))
    scheduler = PolynomialLR(optimizer, total_iters=int(hparams["epochs"]), power=hparams["power"])
    # scheduler = CosineAnnealingLR(optimizer, T_max=hparams["T_max"], eta_min=hparams["eta_min"])

    wandb_group = "completed"

    checkpoint_dir = trial.study.study_name
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    try:
        run = wandb.init(project="DeepONet-Optuna-y", group=wandb_group, config=hparams, reinit=False)
        for epoch in range(hparams["epochs"]):
            train_loss = train_epoch(model, optimizer, dl_train, device)
            val_loss = evaluate(model, dl_val, device)
            scheduler.step()

            trial.report(val_loss, epoch)
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch+1, "lr": scheduler.get_last_lr()[0]})

            progress.update(task_id, advance=1)

            if trial.should_prune():
                wandb_group = "pruned"  # if pruned, change wandb group
                raise optuna.TrialPruned()

        test_loss = evaluate(model, dl_test, device)
        wandb.log({"test_loss": test_loss})

    except optuna.TrialPruned:
        run.finish(exit_code=255)  # if pruned, finish run with error code
        raise

    trial_path = os.path.join(checkpoint_dir, f"trial_{trial.number}.pt")
    torch.save(model.state_dict(), trial_path)
    trial.set_user_attr("checkpoint", trial_path)

    run.finish()

    return val_loss

if __name__ == "__main__":
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    data_path = "data/"
    x, y, grfs, grf_ints, n_samples = load_data(data_path)

    grf_train, grf_val, grf_test, y_train, y_val, y_test, grf_int_train, grf_int_val, grf_int_test = train_val_test_split(n_samples, grfs, y, grf_ints)

    ds_train = IntegralData(grf_train, y_train, grf_int_train)
    ds_val = IntegralData(grf_val, y_val, grf_int_val)
    ds_test = IntegralData(grf_test, y_test, grf_int_test)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # sampler = TPESampler(seed=42)
    sampler = BoTorchSampler(seed=42)

    console = Console()
    progress = Progress()
    # pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=10, interval_steps=1)
    # pruner = WilcoxonPruner(p_threshold=0.2, n_startup_steps=10)
    # pruner = PercentilePruner(25.0, n_startup_trials=10, n_warmup_steps=10, interval_steps=10)
    pruner = HyperbandPruner()

    with progress:
        task_id = progress.add_task("[green]Optuna Trials", total=args.n_trials)

        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner, study_name="DeepONet_Trials", storage="sqlite:///optuna.db", load_if_exists=True)

        study.optimize(lambda trial: objective(trial, "Optuna3", console, progress, task_id),
                       n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Plot optimization history
    fig_history = plot_optimization_history(study)
    # make it log scale
    fig_history.update_yaxes(type="log")
    plot(fig_history, filename='figs/optimization_history.html', auto_open=False)

    fig_importance = plot_param_importances(study)
    plot(fig_importance, filename='figs/param_importances.html', auto_open=False)