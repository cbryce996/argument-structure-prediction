import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNClassifier, self).__init__()

        # GCN layer
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Fully connected layer for edge classification
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply the first GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Apply the second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Edge representation by concatenating or summing node embeddings
        edge_x = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)

        # Fully connected layer for edge classification
        edge_scores = self.fc(edge_x)

        return F.log_softmax(edge_scores, dim=1)