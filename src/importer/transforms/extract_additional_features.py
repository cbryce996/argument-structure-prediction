import networkx as nx
import numpy as np
import torch
from torch_geometric.transforms import BaseTransform
from transformers import pipeline

from utils import ThreadUtils

thread_utils = ThreadUtils()


class ExtractAdditionalFeatures(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data):
        graph = data.graph

        try:
            degree_centrality = nx.degree_centrality(graph)
            nx.set_node_attributes(graph, degree_centrality, name="centrality")
        except Exception as e:
            thread_utils.thread_safe_print(f"Error calculating degree centrality: {e}")

        try:
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            revision = "af0f99b"
            sentiment_pipeline = pipeline(
                "sentiment-analysis", model=model_name, revision=revision
            )
            for node in graph.nodes:
                text = graph.nodes[node].get("text", "")
                try:
                    result = sentiment_pipeline(text)
                    sentiment_score = result[0]["score"]
                    graph.nodes[node]["sentiment"] = sentiment_score
                except Exception as pe:
                    thread_utils.thread_safe_print(
                        f"Error analyzing sentiment for node {node}: {pe}"
                    )
        except Exception as e:
            thread_utils.thread_safe_print(
                f"Error initializing sentiment analysis pipeline: {e}"
            )

        thread_utils.thread_safe_print(
            "Additional features extracted and added to graph as node attributes."
        )

        return data
