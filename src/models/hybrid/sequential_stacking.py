import torch
import torch.nn as nn
import torch.nn.functional as F
from ..single.gcn import GCNModel
from ..single.gat import GATModel
from ..single.gin import GINModel

class SequentialStacking(nn.Modulee):
    def __init__(self, num_features, hidden_size):
        super(SequentialStacking, self).__init__()
        self.gcn = GCNModel(num_features, hidden_size)
        self.gat = GATModel(num_features, hidden_size)
        self.gin = GINModel(num_features, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply GCN layer
        x = self.gcn(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Apply GAT layer
        x = self.gat(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Apply GIN layer
        x = self.gin(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Concatenate node representations for edge prediction
        edge_representation = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)

        # Fully connected layer for link prediction
        edge_scores = self.fc(edge_representation).squeeze()

        return edge_scores