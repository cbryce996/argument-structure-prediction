from torch_geometric.transforms import BaseTransform
from utils import ThreadUtils
import networkx as nx

thread_utils = ThreadUtils()

class KeepLargestConnectedComponent(BaseTransform):
    def __call__(self, data):
        graph = data.graph

        largest_component = max(nx.connected_components(graph), key=len)

        largest_graph = nx.Graph(graph.subgraph(largest_component))

        data.graph = largest_graph

        thread_utils.thread_safe_print(f"Kept largest connected component for {data.name}")

        return data