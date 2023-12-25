import torch
from torch import nn
from torch.nn import functional as F

class MLP(torch.nn.Module):
    def __init__(self,input_dim):
        super(MLP,self).__init__()
        self.model = nn.Sequential(
            #nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, 264),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(264, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
            #nn.ReLU(),
            #nn.Dropout(0.1),
            #nn.Linear(32, 16),
            #nn.ReLU(),
            #nn.Dropout(0.1),
            #nn.Linear(16, 3),
        )
        
        self.double()

    def forward(self, x):
        z = self.model(x)
        return z

        