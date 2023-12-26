import torch
from torch import nn
from torch.nn import functional as F

class MLP(torch.nn.Module):
    def __init__(self,dims = [64,28,3]):
        super(MLP,self).__init__()
        self.layers = []
        for i in range(len(dims)-1):
            self.layers.append(nn.Dropout(0.1))
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            self.layers.append(nn.ReLU())
            
        self.model = nn.Sequential(*self.layers
        )
        self.double()

    def forward(self, x):
        z = self.model(x)
        return z

        