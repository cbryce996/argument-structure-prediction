import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size * 2, 2)
        self.bn1 = nn.BatchNorm1d(hidden_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.9, training=self.training)

        x = self.bn1(x)
        x = F.relu(x)

        edge_representation = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)

        edge_scores = self.fc1(edge_representation)

        return edge_scores