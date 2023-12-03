import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class CustomGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomGNNLayer, self).__init__(aggr='add')  # Use 'add' aggregation for message passing
        self.lin = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix for graph connectivity
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Calculate normalization coefficients for each node
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Perform message passing
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def message(self, x_j, norm):
        # Apply a linear transformation to the input features of neighboring nodes
        return norm.view(-1, 1) * self.lin(x_j)

    def update(self, aggr_out):
        # Apply a non-linear activation function
        return F.relu(aggr_out)

class CustomGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomGNN, self).__init__()
        self.conv1 = CustomGNNLayer(input_dim, hidden_dim)
        self.conv2 = CustomGNNLayer(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        # Apply the first GNN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Apply the second GNN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return x
