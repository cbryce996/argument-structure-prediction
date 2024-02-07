import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GCNModel, GATModel, GINModel

class SequentialStackingModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SequentialStackingModel, self).__init__()
        self.gcn = GCNModel(input_size, hidden_size)
        self.gat = GATModel(input_size, hidden_size, num_heads=1)
        self.gin = GINModel(input_size, hidden_size)
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
        edge_scores = self.fc(edge_representation)

        return edge_scores