import torch.nn as nn
import torch
import tqdm

def train(model, train_loader,val_loader , optimizer, criterion, epochs, device = "cpu", pretrain = ""):
    model.train()
    model.to(device)
    
    train_loss = []
    val_loss = []
    with tqdm.tqdm(range(epochs), unit="Epoch") as lossepoch:
        optimizer.zero_grad()
        lossepoch.set_description(f"Train loss = ..., Val loss = ...")
        for epoch in lossepoch:
            l = 0
            for data in train_loader:  # Iterate in batches over the training dataset.
                 if pretrain == "encoder":
                     out = model.encoder(data[0].to(device))
                     loss_fl = criterion(out, data[1].to(device))
                     #loss_fl = ((data[1].to(device) - out)**2).sum() + model.encoder.kl
                 elif pretrain == "decoder":
                     out = model.decoder(data[1]).to(device)
                     loss_fl = criterion(out, data[0].to(device))
                     #loss_fl = ((data[0].to(device) - out)**2).sum() + model.encoder.kl
                 elif pretrain == "auto_encode":
                     out = model.encoder(data[0].to(device))
                     loss_fl = criterion(out, data[1].to(device))
                     
                     loss_fl.backward(retain_graph=True)  # Derive gradients.
                     l += loss_fl.cpu().detach().numpy()
                     optimizer.step()  # Update parameters based on gradients.
                     optimizer.zero_grad()  # Clear gradients.
                     
                     out = model.decoder(out).to(device)
                     loss_fl = criterion(out, data[0].to(device))
                     loss_fl.backward()  # Derive gradients.
                     l += loss_fl.cpu().detach().numpy()
                     optimizer.step()  # Update parameters based on gradients.
                     optimizer.zero_grad()  # Clear gradients.
                     continue
                 else:
                     out = model(data[0].to(device))
                     loss_fl = criterion(out, data[0].to(device))
                     #loss_fl = ((data[0].to(device) - out)**2).sum() + model.encoder.kl
    
                
                 loss_fl.backward()  # Derive gradients.
                 l += loss_fl.cpu().detach().numpy()
                 optimizer.step()  # Update parameters based on gradients.
                 optimizer.zero_grad()  # Clear gradients.
                 #for p in model.parameters():
                 #   p.data.clamp_(0)
            l = l/len(train_loader)
            train_loss.append(l)
            v = val(model, val_loader, optimizer, criterion, device, pretrain)
            val_loss.append(v)
            lossepoch.set_description(f"Train loss = {l}, Val loss = {v}")
    
    return model, train_loss, val_loss

def val(model, loader, optimizer, criterion, device, pretrain):
     model.eval()
     model.to(device)
     l = 0
     for data in loader:  
         if pretrain == "encoder":
            out = model.encoder(data[0].to(device))
            #out = torch.sin(out*torch.pi) 
            loss_fl = criterion(out, data[1].to(device))
         elif pretrain == "decoder":
            #out = model.decoder(torch.sin(data[1]*torch.pi).to(device))
            out = model.decoder(data[1]).to(device)
            loss_fl = criterion(out, data[0].to(device))
         else:
            out = model(data[0].to(device))
            loss_fl = criterion(out, data[0].to(device))
         
         l += loss_fl.cpu().detach().numpy()
    
     return l/len(loader)

