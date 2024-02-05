from torch_geometric.transforms import BaseTransform
from utils import ThreadUtils

thread_utils = ThreadUtils()

class KeepSelectedNodeTypes(BaseTransform):
    def __init__(self, types_to_keep):
        self.types_to_keep = types_to_keep

    def __call__(self, data):
        graph = data.graph

        try:
            nodes_to_remove = [node for node, attr in graph.nodes(data=True) if attr.get("type") not in self.types_to_keep]
    
            for node in nodes_to_remove:
                graph.remove_node(node)

            data.graph = graph

            thread_utils.thread_safe_print(f"Successfully kept only node types {self.types_to_keep} in {data.name}")

            return data
            
        except Exception as error:
            thread_utils.thread_safe_print(f"Failed to keep only node types {self.types_to_keep} in {data.name}: {str(error)}")