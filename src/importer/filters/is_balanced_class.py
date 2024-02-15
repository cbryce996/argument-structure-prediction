import networkx as nx
from torch_geometric.transforms import BaseTransform

from utils import ThreadUtils

thread_utils = ThreadUtils()


class IsBalancedClass(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data):
        labels = nx.get_edge_attributes(data.graph, "type")
        unique_labels = set(label for label in labels.values())

        if len(unique_labels) >= 3:
            thread_utils.thread_safe_print(
                f"Class balance condition passed for {data.name}"
            )
            return data

        thread_utils.thread_safe_print(
            f"Class balance condition not met for {data.name}"
        )
        return False
