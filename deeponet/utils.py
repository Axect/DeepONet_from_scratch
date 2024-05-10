import torch
import matplotlib.pyplot as plt
import scienceplots
import os

class Predictor:
    def __init__(self, model, device="cpu", study_name="NeuralHamilton-VAE", run_name="trial_87"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.path = f"analysis/{study_name}/{run_name}"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def predict(self, u, y):
        with torch.no_grad():
            u = u.view(1, -1)
            y = y.view(1, -1)
            u = u.to(self.device)
            y = y.to(self.device)
            pred = self.model(u, y)
            return pred
        
    def potential_plot(self, x, u, name="potential"):
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            fig.set_dpi(600)
            ax.autoscale(tight=True)
            ax.plot(x, u)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$u(x)$")
            plt.savefig(f"{self.path}/{name}.png", dpi=600, bbox_inches="tight")

    def predict_plot(self, u, y, Guy, name="prediction"):
        Guy_pred = self.predict(u, y).squeeze(0)

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
            plt.savefig(f"{self.path}/{name}.png", dpi=600, bbox_inches="tight")

class VAEPredictor(Predictor):
    def __init__(self, model, device="cpu", study_name="NeuralHamilton-VAE", run_name="trial_87"):
        super().__init__(model, device, study_name, run_name)
        self.model.reparametrize = False

    def predict(self, u, y):
        with torch.no_grad():
            u = u.view(1, -1)
            y = y.view(1, -1)
            u = u.to(self.device)
            y = y.to(self.device)
            pred, _, _ = self.model(u, y)
            return pred
