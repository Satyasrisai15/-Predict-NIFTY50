import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch_idx, (X, Y) in enumerate(dataloader):
        X = X.to(device)
        Y = Y.to(device)
        
        optimizer.zero_grad()
        
        pred_3m, pred_1y, pred_3y = model(X)
        
        loss_3m = criterion(pred_3m.squeeze(), Y[:, 0])
        loss_1y = criterion(pred_1y.squeeze(), Y[:, 1])
        loss_3y = criterion(pred_3y.squeeze(), Y[:, 2])
        
        loss = loss_3m + loss_1y + loss_3y
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.6f}")
    
    return total_loss / n_batches
