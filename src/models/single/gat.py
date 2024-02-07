import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(
            in_channels=input_size,
            out_channels=hidden_size,
            heads=num_heads,
            dropout=0.6
        )

        self.fc = nn.Linear(hidden_size * 2 * 10, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply GAT convolutional layers
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)

        # Concatenate node representations for edge prediction
        edge_representation = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)

        # Fully connected layer for link prediction
        edge_scores = self.fc(edge_representation)

        return edge_scores
