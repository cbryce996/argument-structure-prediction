import networkx as nx
from torch_geometric.transforms import BaseTransform
from utils import ThreadUtils

thread_utils = ThreadUtils()

class KeepLargestConnectedComponent(BaseTransform):
    def __call__(self, data):
        try:
            graph = data.graph

            largest_component = max(nx.connected_components(graph), key=len)

            largest_graph = nx.Graph(graph.subgraph(largest_component))

            data.graph = largest_graph

            thread_utils.thread_safe_print(f"Kept largest connected component for {data.name}")
        
        except Exception as e:
            thread_utils.thread_safe_print(f'Failed to keep largest connected component for {data.name}: {str(e)}')

        return data
