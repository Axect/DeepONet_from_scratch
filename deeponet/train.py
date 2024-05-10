import torch
import torch.nn.functional as F
import wandb

class Trainer:
    def __init__(self, model, optimizer, scheduler, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def step(self, u, y):
        pred = self.model(u.to(self.device), y.to(self.device))
        return pred

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0
        for u, y, Guy in dataloader:
            self.optimizer.zero_grad()
            pred = self.step(u, y)
            loss = F.mse_loss(pred, Guy.to(self.device))
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        return epoch_loss
    
    def evaluate(self, dataloader):
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            for u, y, Guy in dataloader:
                pred = self.step(u, y)
                loss = F.mse_loss(pred, Guy.to(self.device))
                eval_loss += loss.item()
        eval_loss /= len(dataloader)
        return eval_loss
    
    def train(self, dataloader, progress, epochs=500):
        progress_epoch = progress.add_task("[cyan]Epochs", total=epochs)
        for epoch in range(epochs):
            train_loss = self.train_epoch(dataloader)
            val_loss = self.evaluate(dataloader)
            self.scheduler.step()
            progress.update(progress_epoch, advance=1)
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch+1, 
                "lr": self.scheduler.get_last_lr()[0]
            })
        progress.remove_task(progress_epoch)
            
    
class VAETrainer:
    def __init__(self, model, optimizer, scheduler, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def step(self, u, y):
        pred, mu, logvar = self.model(u.to(self.device), y.to(self.device))
        return pred, mu, logvar
    
    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0
        epoch_kl_loss = 0
        for u, y, Guy in dataloader:
            self.optimizer.zero_grad()
            pred, mu, logvar = self.step(u, y)

            # Flatten
            mu_vec = mu.view((mu.shape[0], -1))             # B, D * L * Z
            logvar_vec = logvar.view((logvar.shape[0], -1))

            # KL Divergence (mean over latent dimensions)
            kl_loss = -0.5 * torch.mean(1 + logvar_vec - mu_vec.pow(2) - logvar_vec.exp(), dim=1)
            kl_loss = self.model.kl_weight * torch.mean(kl_loss)
            loss = F.mse_loss(pred, Guy.to(self.device)) + kl_loss
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
        epoch_loss /= len(dataloader)
        epoch_kl_loss /= len(dataloader)
        return epoch_loss, epoch_kl_loss
    
    def evaluate(self, dataloader):
        self.model.eval()
        eval_loss = 0
        eval_kl_loss = 0
        with torch.no_grad():
            for u, y, Guy in dataloader:
                pred, mu, logvar = self.step(u, y)

                # Flatten
                mu_vec = mu.view((mu.shape[0], -1))             # B, D * L * Z
                logvar_vec = logvar.view((logvar.shape[0], -1))

                # KL Divergence (mean over latent dimensions)
                kl_loss = -0.5 * torch.mean(1 + logvar_vec - mu_vec.pow(2) - logvar_vec.exp(), dim=1)
                kl_loss = self.model.kl_weight * torch.mean(kl_loss)
                loss = F.mse_loss(pred, Guy.to(self.device)) + kl_loss

                eval_loss += loss.item()
                eval_kl_loss += kl_loss.item()
        eval_loss /= len(dataloader)
        eval_kl_loss /= len(dataloader)
        return eval_loss, eval_kl_loss
    
    def train(self, dataloader, progress, epochs=500):
        progress_epoch = progress.add_task("[cyan]Epochs", total=epochs)
        for epoch in range(epochs):
            train_loss, train_kl_loss = self.train_epoch(dataloader)
            val_loss, val_kl_loss = self.evaluate(dataloader)
            self.scheduler.step()
            progress.update(progress_epoch, advance=1)
            wandb.log({
                "train_loss": train_loss,
                "train_kl_loss": train_kl_loss,
                "val_loss": val_loss,
                "val_kl_loss": val_kl_loss,
                "epoch": epoch+1,
                "lr": self.scheduler.get_last_lr()[0]
            })
        progress.remove_task(progress_epoch)

            
