import torch
import numpy as np
from torch_geometric.transforms import BaseTransform
from embeddings import BERTEmbedding, BagOfWordsEmbedding, Word2VecEmbedding, Doc2VecEmbedding
from utils import ThreadUtils

thread_utils = ThreadUtils()

class EmbedNodeText(BaseTransform):
    def __init__(self, embedding_method, max_embedding_length=128):
        self.embedding_method = embedding_method
        self.max_embedding_length = max_embedding_length

    def __call__(self, data):
        graph = data.graph

        node_text = [graph.nodes[node]["text"] for node in graph.nodes()]
        node_embeddings = []

        for text in node_text:
            if self.embedding_method == 'bert-generic':
                embedding_model = BERTEmbedding()
            elif self.embedding_method == 'bag-of-words':
                embedding_model = BagOfWordsEmbedding()
            elif self.embedding_method == 'word2vec':
                embedding_model = Word2VecEmbedding()
            elif self.embedding_method == 'doc2vec':
                embedding_model = Doc2VecEmbedding()
            else:
                raise ValueError(f"Unsupported embedding method: {self.embedding_method}")

            node_embeddings.append(embedding_model.embed_text(text))
        
        for i, node in enumerate(graph.nodes()):
            graph.nodes[node]["embedding"] = node_embeddings[i]

        thread_utils.thread_safe_print(f"Created {self.embedding_method.capitalize()} embeddings for {data.name}")

        data.graph = graph

        return data