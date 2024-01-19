import torch.nn as nn
import torch
import tqdm

def train(model, train_loader,val_loader , optimizer, criterion, epochs, device = "cpu"):
    model.train()
    model.to(device)
    
    train_loss = []
    val_loss = []
    with tqdm.tqdm(range(epochs), unit="Epoch") as lossepoch:
        lossepoch.set_description(f"Train loss = ..., Val loss = ...")
        for epoch in lossepoch:
            l = 0
            for data in train_loader:  # Iterate in batches over the training dataset.
                 out = model(data[0].to(device))
                 loss_fl = criterion(out, data[0].to(device))
                 loss_fl.backward()  # Derive gradients.
                 l += loss_fl.cpu().detach().numpy()
                 optimizer.step()  # Update parameters based on gradients.
                 optimizer.zero_grad()  # Clear gradients.
            l = l/len(train_loader)
            train_loss.append(l)
            v = val(model, val_loader, optimizer, criterion, device)
            val_loss.append(v)
            lossepoch.set_description(f"Train loss = {l}, Val loss = {v}")
    
    return train_loss, val_loss

def val(model, loader, optimizer, criterion, device):
     model.eval()
     model.to(device)
     l = 0
     for data in loader:  
        
         out = model(data[0].to(device)) 
         loss_fl = criterion(out, data[0].to(device))
         l += loss_fl.cpu().detach().numpy()
    
     return l/len(loader)


