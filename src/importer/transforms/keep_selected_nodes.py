import networkx as nx
from torch_geometric.transforms import BaseTransform
from utils import ThreadUtils

thread_utils = ThreadUtils()

class KeepSelectedNodeTypes(BaseTransform):
    def __init__(self, types_to_keep=["I", "RA", "MA", "CA"]):
        self.types_to_keep = types_to_keep

    def __call__(self, data):
        graph = data.graph

        try:
            # Filter nodes based on the types to keep
            nodes_to_remove = []
            for node, attr in graph.nodes(data=True):
                if "type" not in attr or attr["type"] not in self.types_to_keep:
                    nodes_to_remove.append(node)

            # Remove nodes that are not of the specified types
            for node in nodes_to_remove:
                graph.remove_node(node)

            # Remove orphaned edges
            #graph.remove_edges_from(graph.selfloop_edges())

            # Reindex nodes to ensure consecutive node indices
            mapping = {old_id: new_id for new_id, old_id in enumerate(graph.nodes())}
            graph = nx.relabel_nodes(graph, mapping)

            data.graph = graph

            thread_utils.thread_safe_print(f"Successfully kept only node types {self.types_to_keep} in {data.name}")

            return data
            
        except Exception as error:
            thread_utils.thread_safe_print(f"Failed to keep only node types {self.types_to_keep} in {data.name}: {str(error)}")