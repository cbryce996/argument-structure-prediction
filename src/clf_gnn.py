import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# Define a GNN model
class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(row, dtype=x.dtype)
        return self.propagate(edge_index, x=x, deg=deg)

    def message(self, x_j, deg):
        return x_j / deg.view(-1, 1)

    def update(self, aggr_out):
        return F.relu(self.lin(aggr_out))

class GNNEdgeTypeClassifier(nn.Module):
    def __init__(self):
        super(GNNEdgeTypeClassifier, self).__init__()
        self.embedding = nn.Embedding(128, 30)
        self.layers = nn.ModuleList([GNNLayer(30, 30) for _ in range(3)])
        self.lin = nn.Linear(30, 5)
        self.num_edge_types = 5

    def forward(self, x, edge_index):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.lin(x)
        return x