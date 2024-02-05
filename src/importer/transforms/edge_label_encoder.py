import torch
from torch_geometric.transforms import BaseTransform
from utils import thread_safe_print

class EdgeLabelEncoder(BaseTransform):
    def __init__(self):
        self.label_to_index = {}
        self.index_to_label = {}
        self.num_labels = 0

    def __call__(self, data):
        graph = data.graph

        unique_labels_before = set([graph.edges[src, dst].get("type") for src, dst in graph.edges()])
        print(f"Unique edge labels before encoding: {unique_labels_before}")

        unique_labels = set([graph.edges[src, dst].get("type") for src, dst in graph.edges()])
        
        for label in unique_labels:
            if label not in self.label_to_index:
                self.label_to_index[label] = self.num_labels
                self.index_to_label[self.num_labels] = label
                self.num_labels += 1

        label_index = [self.label_to_index[graph.edges[src, dst].get("type")] for src, dst in graph.edges()]
        label_index = torch.tensor(label_index, dtype=torch.long)

        data.edge_labels = label_index
        data.num_unique_labels = len(unique_labels)

        thread_safe_print(f'Encoded edge labels for {data.name}')

        return data