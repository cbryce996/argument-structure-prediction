import torch
from torch_geometric.transforms import BaseTransform

from embeddings import BERTEmbedding
from utils import ThreadUtils

thread_utils = ThreadUtils()


class EmbedNodeText(BaseTransform):
    def __init__(self, max_embedding_length=128):
        self.max_embedding_length = max_embedding_length

    def __call__(self, data):
        graph = data.graph

        node_text = [graph.nodes[node]["text"] for node in graph.nodes()]
        node_embeddings = []

        try:
            embedding_model = BERTEmbedding()
            for text in node_text:
                embedding = embedding_model.embed_text(text)
                node_embeddings.append(embedding)

            for i, node in enumerate(graph.nodes()):
                graph.nodes[node]["embedding"] = node_embeddings[i]

            thread_utils.thread_safe_print(f"Created BERT embeddings for {data.name}")

        except Exception as e:
            thread_utils.thread_safe_print(
                f"Failed to create BERT embeddings for {data.name}: {str(e)}"
            )

        data.graph = graph

        return data
