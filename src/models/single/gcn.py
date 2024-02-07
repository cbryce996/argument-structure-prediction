import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.fch = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        edge_scores = self.fch(x)

        # Concatenate node representations for edge prediction
        edge_representation = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)

        edge_scores = self.fc(edge_representation)

        # Fully connected layer for link prediction
        edge_scores = self.fc(edge_representation)

        return edge_scores