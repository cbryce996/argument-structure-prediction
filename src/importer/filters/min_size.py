import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import is_undirected
from utils import ThreadUtils

thread_utils = ThreadUtils()

class MinNodesAndEdges(BaseTransform):
    def __init__(self, min_nodes=10, min_edges=7):
        self.min_nodes = min_nodes
        self.min_edges = min_edges

    def __call__(self, data):
        num_nodes = data.num_nodes
        num_edges = data.num_edges

        if (num_nodes >= self.min_nodes) and (num_edges >= self.min_edges):
            thread_utils.thread_safe_print(f"Size filter passed for {data.name}")
            return data

        return False