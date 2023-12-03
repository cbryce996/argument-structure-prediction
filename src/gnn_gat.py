import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GATClassifier, self).__init__()

        # GAT layer with multiple heads
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.6)
        
        # Fully connected layer for edge classification
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply the first GAT layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)

        # Apply the second GAT layer
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)

        # Edge representation by concatenating or summing node embeddings
        edge_x = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)

        # Fully connected layer for edge classification
        edge_scores = self.fc(edge_x)

        return F.log_softmax(edge_scores, dim=1)