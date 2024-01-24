import torch
from torch import nn
#device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

class Sin(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(input*torch.pi)

class VENCODER_MLP(torch.nn.Module):
    def __init__(self,dims = [64,2]):
        super(VENCODER_MLP, self).__init__()

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0
        
        self.encoder1 = []
        self.encoder1.append(nn.Dropout(0.1))
        self.encoder1.append(nn.Linear(dims[-2], dims[-1]))
        self.encoder1.append(nn.ReLU())
        self.encoder1 = nn.Sequential(*self.encoder1)



        self.encoder2 = []
        self.encoder2.append(nn.Dropout(0.1))
        self.encoder2.append(nn.Linear(dims[-2], dims[-1]))
        self.encoder2.append(Sin())
        self.encoder2 = nn.Sequential(*self.encoder2)

    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mu = self.encoder2(z)
        sigmaballs = self.encoder1(z)

        z = mu + sigmaballs*self.N.sample(mu.shape)
        self.kl = (sigmaballs**2 + mu**2 - torch.log(sigmaballs) - 1/2).sum()
        return z



class VAE_MLP(torch.nn.Module):
    def __init__(self,dims = [64,2]):
        super(VAE_MLP, self).__init__()

        self.encoder = VENCODER_MLP(dims)

        # reverse
        dims = dims[::-1]
        self.decoder = []
        self.decoder.append(nn.Dropout(0.1))
        self.decoder.append(nn.Linear(dims[-2], dims[-1]))
        self.decoder = nn.Sequential(*self.decoder
        )
        
        self.float()
        

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        
        z = self.encoder(z)
        z = self.decoder(z)
        
        return z

        