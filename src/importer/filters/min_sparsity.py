from torch_geometric.transforms import BaseTransform
from utils import ThreadUtils
import networkx as nx

thread_utils = ThreadUtils()

class MinSparsityAndConnectivity(BaseTransform):
    def __init__(self, sparsity_threshold=0.5):
        self.sparsity_threshold = sparsity_threshold

    def __call__(self, data):
        graph = data.graph

        connectivity_condition = self.is_connected(graph)
        thread_utils.thread_safe_print(f"Connectivity Condition: {connectivity_condition}")

        sparsity_condition = self.calculate_sparsity(graph) > self.sparsity_threshold
        thread_utils.thread_safe_print(f"Sparsity Condition: {sparsity_condition}")

        result = connectivity_condition and sparsity_condition
        thread_utils.thread_safe_print(f"Overall Result: {result}")

        if result:
            thread_utils.thread_safe_print(f"Sparsity and Connectivity filter passed for {data.name}")
            return data

        return False

    def is_connected(self, graph):
        return nx.is_connected(graph)

    def calculate_sparsity(self, graph):
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        max_edges = (num_nodes * (num_nodes - 1)) // 2

        sparsity = 1.0 - (num_edges / max_edges)
        return sparsity
