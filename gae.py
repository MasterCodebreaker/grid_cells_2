

class GCN(torch.nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()

        amp_size = 1000
        
        self.conv1 = GATConv(data_list[0].num_node_features, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim)
        self.conv4 = GATConv(hidden_dim, hidden_dim)
        self.conv5 = GATConv(hidden_dim, hidden_dim)
        self.conv6 = GATConv(hidden_dim, hidden_dim)
        self.conv7 = GATConv(hidden_dim, data_list[0].num_node_features)
        
        self.amp = AdaptiveMaxPool1d(amp_size)
        
        self.linear1 = torch.nn.Linear(amp_size, amp_size)
        
        self.linear2 = torch.nn.Linear(amp_size, 272*batch_size)

        self.float()

    def forward(self, data):
        og_size = data.y.shape[0]

        data.x = self.conv1(data.x, data.edge_index)
        cluster = graclus_cluster(data.edge_index[0], data.edge_index[1], num_nodes = data.x.shape[0])
        data = max_pool(cluster, data)

        """
        data.x = self.conv2( data.x , data.edge_index)
        cluster = graclus_cluster(data.edge_index[0], data.edge_index[1], num_nodes = data.x.shape[0])
        data = max_pool(cluster, data)

        data.x = self.conv3( data.x , data.edge_index)
        cluster = graclus_cluster(data.edge_index[0], data.edge_index[1], num_nodes = data.x.shape[0])
        data = max_pool(cluster, data)

        data.x = self.conv4( data.x , data.edge_index)
        cluster = graclus_cluster(data.edge_index[0], data.edge_index[1], num_nodes = data.x.shape[0])
        data = max_pool(cluster, data)

        data.x = self.conv5( data.x , data.edge_index)
        cluster = graclus_cluster(data.edge_index[0], data.edge_index[1], num_nodes = data.x.shape[0])
        data = max_pool(cluster, data)

        data.x = self.conv6( data.x , data.edge_index)
        cluster = graclus_cluster(data.edge_index[0], data.edge_index[1], num_nodes = data.x.shape[0])
        data = max_pool(cluster, data)

        data.x = self.conv7( data.x , data.edge_index)
        cluster = graclus_cluster(data.edge_index[0], data.edge_index[1], num_nodes = data.x.shape[0])
        data = max_pool(cluster, data)
        #print(data)
        """
        
        x = data.x
        
        x = torch.flatten(x)
        x = x[None,:]
        x = x.squeeze(-1)
        x = self.amp(x)

    
        x = self.linear1(x)
        x = ReLU()(x)

        x = self.linear2(x)
        x = ReLU()(x)
        
        x = x.t()
        x = x[:og_size,:]

        #print(x.shape)

        return x #F.log_softmax(x, dim=1)