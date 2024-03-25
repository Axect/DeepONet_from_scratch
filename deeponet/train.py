import torch
import torch.nn.functional as F

def train_epoch(model, optimizer, dataloader, device):
    model.train()
    epoch_loss = 0
    for u, y, Guy in dataloader:  
        optimizer.zero_grad()
        pred = model(u.to(device), y.to(device))
        loss = F.mse_loss(pred, Guy.to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(dataloader)
    return epoch_loss

def evaluate(model, dataloader, device):
    model.eval()  
    eval_loss = 0
    with torch.no_grad():
        for u, y, Guy in dataloader:
            pred = model(u.to(device), y.to(device))
            loss = F.mse_loss(pred, Guy.to(device)) 
            eval_loss += loss.item()
    eval_loss /= len(dataloader)
    return eval_loss