import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import add_self_loops, degree

class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return F.relu(self.lin(aggr_out))

class GNNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GNNClassifier, self).__init__()
        self.embedding = nn.Linear(768, in_channels)  # Adjust in_channels based on BERT embedding size
        self.layers = nn.ModuleList([
            GNNLayer(in_channels, hidden_channels),
            GNNLayer(hidden_channels, hidden_channels),
            GNNLayer(hidden_channels, hidden_channels)
        ])
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.embedding(x)  # Assuming x is the BERT node embeddings
        for layer in self.layers:
            x = layer(x, edge_index)
        x = global_add_pool(x, batch=None)  # Global pooling over nodes
        x = F.relu(x)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)