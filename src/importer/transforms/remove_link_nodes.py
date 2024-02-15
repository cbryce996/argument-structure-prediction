from torch_geometric.transforms import BaseTransform

from utils import ThreadUtils

thread_utils = ThreadUtils()


class RemoveLinkNodeTypes(BaseTransform):
    def __init__(self, types_to_remove=["RA", "MA", "CA"]):
        self.types_to_remove = types_to_remove

    def __call__(self, data):
        graph = data.graph

        try:
            link_nodes = [
                node
                for node, attrs in graph.nodes(data=True)
                if attrs.get("type") in self.types_to_remove is not None
            ]

            for node in link_nodes:
                edges_to_add = []

                for neighbor in graph.neighbors(node):
                    for successor in graph.neighbors(node):
                        if neighbor != successor and not graph.has_edge(
                            neighbor, successor
                        ):
                            edges_to_add.append(
                                (neighbor, successor, graph.nodes[node]["type"])
                            )

                graph.remove_node(node)

                for source, target, link_type in edges_to_add:
                    graph.add_edge(source, target, type=link_type)

            data.graph = graph

            thread_utils.thread_safe_print(
                f"Successfully removed link node types {self.types_to_remove} in {data.name}"
            )

            return data

        except Exception as error:
            thread_utils.thread_safe_print(
                f"Failed to remove link node types {self.types_to_remove} in {data.name}: {str(error)}"
            )
