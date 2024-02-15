import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GATConv(input_size, hidden_size, heads=4)
        self.fc1 = nn.Linear(hidden_size * 4, 3)
        self.dropout_rate = dropout_rate

        # Attention mechanism parameters
        self.attention = nn.Linear(hidden_size, 3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        edge_representation = (x[edge_index[0]] * x[edge_index[1]]) / 2

        edge_scores = self.fc1(edge_representation)

        return edge_scores
