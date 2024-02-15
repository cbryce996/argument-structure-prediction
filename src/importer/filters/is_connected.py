import networkx as nx
from torch_geometric.transforms import BaseTransform

from utils import ThreadUtils

thread_utils = ThreadUtils()


class IsConnected(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data):
        graph = data.graph
        connectivity_condition = nx.is_connected(graph)

        if connectivity_condition:
            thread_utils.thread_safe_print(
                f"Connectivity filter passed for {data.name}"
            )
            return data

        return False
