import networkx as nx
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
            nx.set_node_attributes(graph, degree_centrality, name="degree_centrality")

            betweenness_centrality = nx.betweenness_centrality(graph)
            nx.set_node_attributes(
                graph, betweenness_centrality, name="betweenness_centrality"
            )

            closeness_centrality = nx.closeness_centrality(graph)
            nx.set_node_attributes(
                graph, closeness_centrality, name="closeness_centrality"
            )

            clustering_coefficient = nx.clustering(graph)
            nx.set_node_attributes(
                graph, clustering_coefficient, name="clustering_coefficient"
            )
        except Exception as e:
            thread_utils.thread_safe_print(f"Error calculating graph features: {e}")

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
