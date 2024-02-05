import torch
from torch_geometric.transforms import BaseTransform
from utils import ThreadUtils

thread_utils = ThreadUtils()

class GraphToPyG(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data):
        nodes = {node: data.graph.nodes[node] for node in data.graph.nodes}
        edges = {edge: data.graph.edges[edge] for edge in data.graph.edges}

        node_mapping = {node: idx for idx, node in enumerate(nodes)}

        node_embeddings = [nodes[node]['embedding'] for node in data.graph.nodes]
        node_embeddings_tensor = torch.tensor([embeddings for embeddings in node_embeddings])
        flattened_embeddings = node_embeddings_tensor.view(node_embeddings_tensor.size(0), -1)
        data.x = flattened_embeddings

        edge_index = [(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in edges]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        data.edge_index = edge_index

        data.y = data.edge_labels

        thread_utils.thread_safe_print(f'Converted graph data to PyG data for {data.name}')

        return data