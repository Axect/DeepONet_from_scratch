import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import scienceplots

def create_activation(activation_name):
    if activation_name == 'ReLU':
        return nn.ReLU()
    elif activation_name == 'GELU':
        return nn.GELU()
    elif activation_name == 'SiLU':
        return nn.SiLU()
    elif activation_name == 'Mish':
        return nn.Mish()
    else:
        raise ValueError(f"Unsupported activation: {activation_name}")
    
def predict(model, u, y):
    with torch.no_grad():
        u = u.view(1, -1)
        y = y.view(1, -1)
        pred = model(u, y)
        return pred
    
def predict_plot(model, u, y, Guy, path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    u = u.view(1, -1)
    y = y.view(1, -1)
    model  = model.to(device)
    u = u.to(device)
    y = y.to(device)
    Guy_pred = predict(model, u, y).squeeze(0)

    u = u.detach().cpu().numpy()
    y = y.detach().cpu().numpy().reshape(-1)
    Guy = Guy.detach().cpu().numpy().reshape(-1)
    Guy_pred = Guy_pred.detach().cpu().numpy().reshape(-1)
    
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        fig.set_dpi(600)
        ax.autoscale(tight=True)
        ax.plot(y, Guy, 'r--', label="Exact", alpha=0.6)
        ax.plot(y, Guy_pred, 'g-.', label="Predicted", alpha=0.6)
        ax.set_xlabel(r"$y$")
        ax.set_ylabel(r"$G(u)(y)$")
        ax.legend()
        if path is None:
            plt.show()
        else:
            plt.savefig(path, dpi=600, bbox_inches="tight")
