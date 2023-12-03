import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class SAGEClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SAGEClassifier, self).__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.fc = nn.Linear(hidden_dim*2, output_dim)  # Concatenate node embeddings for edge representation

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
        edge_x = torch.cat([x[edge_index[0]],x[edge_index[1]]], dim=-1)

        # Fully connected layer for edge classification
        edge_scores = self.fc(edge_x)

        return F.log_softmax(edge_scores, dim=1)