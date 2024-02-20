from torch_geometric.transforms import BaseTransform

from utils import ThreadUtils

thread_utils = ThreadUtils()


class MinSparsity(BaseTransform):
    def __init__(self, sparsity_threshold=0.5):
        self.sparsity_threshold = sparsity_threshold

    def __call__(self, data):
        graph = data.graph
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        max_edges = (num_nodes * (num_nodes - 1)) // 2
        sparsity = 1.0 - (num_edges / max_edges)

        sparsity_condition = sparsity > self.sparsity_threshold

        if sparsity_condition:
            thread_utils.thread_safe_print(f"Sparsity filter passed for {data.name}")
            return data

        return False
