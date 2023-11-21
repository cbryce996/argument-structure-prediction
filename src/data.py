import os
import torch
import json
import networkx as nx
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import BaseTransform
from bert_embd import get_bert_embeddings
from data_load import load_data_from_json

class AIFData(Data):
    """
    Custom class for representing AIF graphs with PyG.

    Attributes:
    - x: Node text embeddings
    - edge_index: Edge index for graph connectivity
    - y: Feature labels
    - graph: Networkx graph
    """
    def __init__(self, x=None, edge_index=None, y=None, graph=None, **kwargs):
        super(AIFData, self).__init__(x=x, edge_index=edge_index, y=y, **kwargs)
        self.graph = graph

    def process_json(self, data):
        """
        Process JSON data and create a Networkx graph representation.

        Args:
        - data: JSON data representing the graph

        Returns:
        - None
        """
        graph = nx.DiGraph()

        try:
            for i, node in enumerate(data["nodes"]):
                graph.add_node(node["nodeID"], type=node["type"], text=node["text"])
            
            for edge in data["edges"]:
                graph.add_edge(edge["fromID"], edge["toID"])
    
            self.graph = graph
            print(self.graph)
        
        except Exception as e:
            print(f"An error occurred while processing data: {str(e)}")
            self.graph = graph
    
    def update_tensors(self):
        graph = self.graph

        self.x = torch.stack([graph.nodes[node]["text_embedding"] for node in graph.nodes()])
        # TODO Rest of PyG data


class AIFDataset(Dataset):
    """
    Custom class for representing a set of AIF graphs.

    Attributes:
    - root: Folder with AIF json files

    Description:
    - Processes and saves the imported and transformed AIF json files
    """
    def __init__(self, root, transform=None, pre_transform=None):
        super(AIFDataset, self).__init__(root, transform, pre_transform)
            
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if hasattr(self, 'processed_dir'):
            processed_files = [f for f in os.listdir(self.processed_dir) if os.path.isfile(os.path.join(self.processed_dir, f))]
            return processed_files
        else:
            return []

    def download(self):
        pass

    def process(self):
        data_list = []
        for root, _, files in os.walk(self.root):
            for file_name in files:
                if file_name.endswith('.json'):
                    json_file_path = os.path.join(root, file_name)
                    print(f"Processing {json_file_path}")
                    graph = load_data_from_json(json_file_path)
                    if graph is not None:
                        data = AIFData()
                        data.process_json(graph)
                        data_list.append(data)

        if hasattr(self, 'processed_dir'):
            processed_data_path = os.path.join(self.processed_dir, 'processed_data.pth')
            torch.save(data_list, processed_data_path)
            print(f"Processed data saved at {processed_data_path}")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(self.processed_paths[idx])
    
class CreateBertEmbeddings(BaseTransform):
    """
    Custom transform for creating BERT embeddings from node text.

    Description:
    - Processes and adds padded BERT embeddings to the graph

    Args:
    - tokenizer: BERT tokenizer
    - model: BERT model
    - max_seq_length: Maximum sequence length for BERT input

    Returns:
    - Transformed PyG Data object
    """
    def __init__(self, tokenizer, model, max_seq_length=128):
        self.tokenizer = tokenizer
        self.model = model
        self.max_seq_length = max_seq_length

    def __call__(self, data):
        graph = data.graph

        texts = [graph.nodes[node]["text"] for node in graph.nodes()]


        tokenized_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )

        attention_mask = tokenized_inputs["attention_mask"]

        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)
            embeddings = outputs.last_hidden_state

        for i, node in enumerate(graph.nodes()):
            graph.nodes[node]["text_embedding"] = embeddings[i].tolist()
            graph.nodes[node]["attention_mask"] = attention_mask[i].tolist()

        return data
    
class RemoveLinkNodes(BaseTransform):
    """
    Custom transform for removing link nodes from a graph.

    Returns:
    - Transformed PyG Data object with link nodes removed
    """
    def __call__(self, data):
        graph = data.graph
        link_node_types = ["YA", "RA", "MA", "TA", "CA"]

        try:
            link_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get("type") in link_node_types]

            # Remove link nodes
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

        except Exception as e:
            print(f"An error occurred while processing data: {str(e)}")
        
        data.graph = graph
        print(f'Graph with link nodes removes: {data.graph}')

        return data
    
class BinaryEdgeLabelEncoder(BaseTransform):
    """
    Custom transform for creating binary links in the graph.

    Attributes:
    - label_to_index: A dictionary mapping edge labels to numerical indices
    - index_to_label: A dictionary mapping numerical indices to edge labels
    - num_labels: The total number of unique edge labels

    Returns:
    - Transformed PyG Data object with binary links.
    """
    def __init__(self):
        self.label_to_index = {'No-Relation': 0, 'Relation': 1}
        self.index_to_label = {0: 'No-Relation', 1: 'Relation'}
        self.num_labels = 2

    def __call__(self, data):
        graph = data.graph

        # Define the mapping for node types to edge labels
        type_to_label_mapping = {0: 'No-Relation', 1: 'Relation'}

        all_node_pairs = [(source, target) for source in graph.nodes for target in graph.nodes if source != target]

        for source, target in all_node_pairs:
            if not graph.has_edge(source, target):
                graph.add_edge(source, target, label=type_to_label_mapping[0])
            else:
                graph.edges[source, target]["label"] = type_to_label_mapping[1]
        
        return data
    
class EdgeLabelEncoder(BaseTransform):
    """
    Custom transform for encoding edge labels into numerical encodings.

    Attributes:
    - label_to_index: A dictionary mapping edge labels to numerical indices
    - index_to_label: A dictionary mapping numerical indices to edge labels
    - num_labels: The total number of unique edge labels

    Returns:
    - Transformed PyG Data object with edge labels encoded
    """
    def __init__(self):
        self.label_to_index = {}
        self.index_to_label = {}
        self.num_labels = 0

    def __call__(self, data):
        graph = data.graph

        # Create a set of unique edge labels
        unique_labels = set([graph.edges[src, dst].get("label") for src, dst in graph.edges()])
        
        # Assign numerical encodings to edge labels
        for label in unique_labels:
            if label not in self.label_to_index:
                self.label_to_index[label] = self.num_labels
                self.index_to_label[self.num_labels] = label
                self.num_labels += 1

        # Encode edge labels in the graph
        for src, dst in graph.edges():
            label = graph.edges[src, dst].get("label")
            index = self.label_to_index[label]
            graph.edges[src, dst]["label_encoded"] = index

        return data
    
class EdgeLabelDecoder(BaseTransform):
    """
    Custom transform for decoding numerical edge label encodings back to original edge labels.

    Attributes:
    - label_encoder: An instance of the EdgeLabelEncoder used for encoding edge labels

    Returns:
    - Transformed PyG Data object with numerical edge labels decoded
    """
    def __init__(self, label_encoder):
        self.label_encoder = label_encoder

    def __call__(self, data):
        graph = data.graph

        # Decode edge label encodings in the graph
        for src, dst in graph.edges():
            index = graph.edges[src, dst].get("label_encoded")
            label = self.label_encoder.index_to_label.get(index)
            graph.edges[src, dst]["label_decoded"] = label

        return data