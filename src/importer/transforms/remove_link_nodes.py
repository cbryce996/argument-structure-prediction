from torch_geometric.transforms import BaseTransform
from utils import ThreadUtils

thread_utils = ThreadUtils()

class RemoveLinkNodeTypes(BaseTransform):
    def __init__(self, types_to_remove):
        self.types_to_remove = types_to_remove

    def __call__(self, data):
        graph = data.graph

        try:
            link_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get("type") in self.types_to_remove]

            for node in link_nodes:
                incoming_edges = list(graph.in_edges(node))
                outgoing_edges = list(graph.out_edges(node))

                for in_edge in incoming_edges:
                    for out_edge in outgoing_edges:
                        source_i, _ = in_edge
                        _, target_i = out_edge

                        link_type = graph.nodes[node]["type"]

                        graph.add_edge(source_i, target_i, type=link_type)

                graph.remove_node(node)

            data.graph = graph

            thread_utils.thread_utilsthread_safe_print(f"Successfully removed link node types {self.types_to_remove} in {data.name}")

            return data

        except Exception as error:
            thread_utils.thread_safe_print(f"Failed to remove link node types {self.types_to_remove} in {data.name}: {str(error)}")