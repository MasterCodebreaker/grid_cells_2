import torch.nn as nn
import torch

def train(model, loader, optimizer, criterion):
    model.train()
    train_loss = []

    for data in loader:  # Iterate in batches over the training dataset.
         out = model(data[0])
        
         loss_fl = criterion(out, data[1])
         loss_fl.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.
         train_loss.append(loss_fl)
    return float(torch.mean(torch.tensor([float(i.detach().numpy()) for i in train_loss])).detach())/len(loader)

def val(model, loader, optimizer, criterion):
     model.eval()
     val_loss = []
     for data in loader:  
        
         out = model(data[0]) 
         loss_fl = criterion(out, data[1])
         val_loss.append(loss_fl)
     return float(torch.mean(torch.tensor([float(i.detach().numpy()) for i in val_loss])).detach())/len(loader)  # Derive ratio of correct predictions.


