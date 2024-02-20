import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.3):
        super(GCNModel, self).__init__()
        self.conv1 = GATConv(input_size, hidden_size, heads=2)
        self.fc1 = nn.Linear(hidden_size * 2 * 2, 3)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        edge_representation = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)

        edge_scores = self.fc1(edge_representation)

        return edge_scores
