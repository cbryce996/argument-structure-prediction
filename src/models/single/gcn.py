import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # No change in size
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # Downsize by half
        self.fc3 = nn.Linear(hidden_size // 2, 3)  # Adjust input size accordingly
        self.dropout = nn.Dropout(p=0.5)  # Adjust dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        edge_representation = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)

        x = self.fc1(edge_representation)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        edge_scores = self.fc3(x)
        
        return edge_scores
