from torch_geometric.transforms import BaseTransform

from utils import ThreadUtils

thread_utils = ThreadUtils()


class MinNumberNodes(BaseTransform):
    def __init__(self, min_nodes=5):
        self.min_nodes = min_nodes

    def __call__(self, data):
        num_nodes = data.graph.number_of_nodes()

        if num_nodes >= self.min_nodes:
            thread_utils.thread_safe_print(f"Size filter passed for {data.name}")
            return data

        return False
