import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


class GINModel(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(num_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            ),
            train_eps=True,
        )

        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        edge_representation = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)

        # Fully connected layer for link prediction
        edge_scores = self.fc(edge_representation).squeeze()

        return edge_scores
