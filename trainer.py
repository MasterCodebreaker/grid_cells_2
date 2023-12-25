import torch.nn as nn
import torch

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.L1Loss()

def train(loader):
    model.train()
    train_loss = []

    for data in loader:  # Iterate in batches over the training dataset.
         out = model(data[0])
         #loss_fl = criterion(out, data.y)  # Compute the loss.
         loss_fl = criterion(out, data[1])
         loss_fl.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.
         train_loss.append(loss_fl)
    return float(torch.mean(torch.tensor([float(i.detach().numpy()) for i in train_loss])).detach())/len(loader)

def val(loader):
     model.eval()
     val_loss = []

     #loss_fl = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         #if len(data) == 272*32:
         #out = model(data)
         out = model(data[0]) 
         #loss_fl = criterion(out, data.y)
         loss_fl = criterion(out, data[1])# Use the class with highest probability.
         val_loss.append(loss_fl)
     return float(torch.mean(torch.tensor([float(i.detach().numpy()) for i in val_loss])).detach())/len(loader)  # Derive ratio of correct predictions.


train_loss = []
test_loss = []
N_EPOCH = 50
for epoch in range(1, N_EPOCH):
    train_loss.append(train(train_loader))
    test_loss.append(val(test_loader))
    print(f'Epoch: {epoch:03d}, Train loss: {train_loss[-1]:}, Test loss: {test_loss[-1]:}')
    