import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_fc_layers):
        super(GCNClassifier, self).__init__()

        self.gcn_input = GCNConv(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)

        self.fc_layers = nn.ModuleList()
        for _ in range(num_fc_layers):
            self.fc_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
            ))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gcn_input(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = F.relu(x)
            x = self.dropout(x)

        edge_x = (x[edge_index[0]] + x[edge_index[1]]) / 2

        for fc_layer in self.fc_layers:
            # Note: Use edge_x instead of x in this loop
            edge_x = fc_layer(edge_x)
            edge_x = F.relu(edge_x)
            edge_x = self.dropout(edge_x)

        edge_scores = self.output_layer(edge_x)

        return F.log_softmax(edge_scores, dim=1)