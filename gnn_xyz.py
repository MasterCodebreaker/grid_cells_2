import numpy as np
import torch
from scipy.sparse import csc_matrix
import torch.nn as nn
from torch.nn.modules.linear import Linear
from torch.nn import ReLU
from torch.utils.data import Subset
import torch_cluster
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import max_pool
from torch_cluster import graclus_cluster
from torch.nn import AdaptiveMaxPool1d
from torch_scatter import scatter_add, scatter_mean
from sklearn.cluster import AgglomerativeClustering

class GCN_reg(torch.nn.Module):
    def __init__(self,input_dim, hidden_dim, batch_size, n_nodes):
        super(GCN_reg,self).__init__()

        self.amp_size = hidden_dim#*input_dim
        self.layers = [GATConv(input_dim, hidden_dim)]
        go_too = 3
        for i in range(1,go_too):
            self.layers += [GCNConv(hidden_dim*(i), hidden_dim*(i+1))]
        self.layers +=  [GCNConv(hidden_dim*go_too, hidden_dim*(go_too+1))]
        self.layers = nn.ModuleList(self.layers)

        #self.amp = AdaptiveMaxPool1d(amp_size)
        
                         #self.linear1 = torch.nn.Linear(amp_size, 64)
        
        self.linearx = torch.nn.Linear(self.amp_size*(go_too+1), 3)
        #self.lineary = torch.nn.Linear(self.amp_size, batch_size)
        #self.linearz = torch.nn.Linear(self.amp_size, batch_size)

        #self.float()
        self.double()

    def forward(self, data):
        og_size = data.y.shape[0]
        x = data.x
        alpha = torch.zeros(self.amp_size,data.y.shape[0])
        for l in self.layers:
            #print(data.edge_attr.shape)
            x = l(x, data.edge_index,data.edge_attr.squeeze())
            x = torch.nn.Tanh()(x)
            v=  scatter_mean(x.t(), data.batch)
            #print(v.shape)
            #print(alpha.shape)
            #alpha += scatter_mean(x.t(), data.batch)

            #cluster = graclus_cluster(data.edge_index[0], data.edge_index[1], num_nodes = data.x.shape[0])
            #data = max_pool(cluster, data)

        #x = data.x
        
        #print("before ", x.shape)
        x = scatter_mean(x.t(), data.batch)
        #print(x.shape)
        #x = torch.flatten(x)
        #x = x[None,:]
        #x = x.squeeze(-1)
        #x = self.amp(x)

    
        #x = self.linear1(x)
        #x1 = torch.nn.LeakyReLU()(x)
        #print(x.shape)
        #if x.shape[-1] < self.amp_size:
        #    x = torch.cat((x,torch.zeros((self.amp_size - x.shape[-1]))),0)
        #print(x.t().shape, " ", alpha.shape)


        x = x.t()#+alpha.t()
        
        out = self.linearx(x)[None,:]
        #y = self.lineary(x1)[None,:]
        #z = self.linearz(x1)[None,:]


        #out = torch.cat((x, y, z), 0).t()
        #x = ReLU()(x)
        
        #x = x.t()
        #print(out.shape)
        #print(data.y.shape)
        out = out[:og_size,:].squeeze(0)

        #print(x.shape)

        return out #F.log_softmax(x, dim=1)
