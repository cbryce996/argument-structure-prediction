import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import is_undirected
from torch_sparse import coalesce

class ComplexityFilter(BaseTransform):
    def __init__(self, min_nodes=10, max_nodes=1000, min_edges=5, max_edges=500, connectivity_threshold=0.5, sparsity_threshold=0.1):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_edges = min_edges
        self.max_edges = max_edges
        self.connectivity_threshold = connectivity_threshold
        self.sparsity_threshold = sparsity_threshold

    def __call__(self, data):
        if not isinstance(data, Data):
            raise ValueError("Input must be a PyTorch Geometric Data object.")

        num_nodes = data.num_nodes
        num_edges = data.num_edges
        adjacency = data.adjacency()

        # Assess size
        if not (self.min_nodes <= num_nodes <= self.max_nodes) or not (self.min_edges <= num_edges <= self.max_edges):
            return False

        # Assess connectivity
        if not self.is_connected(adjacency, num_nodes):
            return False

        # Assess sparsity
        sparsity = self.calculate_sparsity(adjacency)
        if sparsity > self.sparsity_threshold:
            return False

        return True

    def is_connected(self, adjacency, num_nodes):
        if not is_undirected(adjacency):
            raise ValueError("Graph must be undirected for connectivity assessment.")
        
        row, col, _ = coalesce(adjacency, torch.zeros(1, dtype=torch.long, device=adjacency.device), num_nodes, num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        connected_components = torch.zeros(num_nodes, dtype=torch.long, device=adjacency.device)
        current_label = 1

        for node in range(num_nodes):
            if connected_components[node] == 0:
                self._dfs(node, current_label, edge_index, connected_components)
                current_label += 1

        return current_label == 2

    def _dfs(self, node, label, edge_index, connected_components):
        connected_components[node] = label
        neighbors = edge_index[1, edge_index[0] == node]
        for neighbor in neighbors.tolist():
            if connected_components[neighbor] == 0:
                self._dfs(neighbor, label, edge_index, connected_components)

    def calculate_sparsity(self, adjacency):
        num_edges = adjacency.size(1)
        num_nodes = adjacency.size(0)
        max_edges = (num_nodes * (num_nodes - 1)) // 2

        sparsity = 1.0 - (num_edges / max_edges)
        return sparsity