import json
import os
import string
import threading
import concurrent.futures as cf
import torch
import networkx as nx
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from importer.transforms import EdgeLabelEncoder
from utils import ThreadUtils

thread_utils = ThreadUtils()

print_lock = threading.Lock()
file_lock = threading.Lock()

class AIFData(Data):
    def __init__(self, x=None, y=None, graph=None, name=None, num_nodes=None):
        super(AIFData, self).__init__(x=x, y=y)
        self.graph = graph
        self.name = name
        self.num_nodes = num_nodes

class AIFDataset(InMemoryDataset):
    label_to_index = {}
    index_to_label = {}
    num_labels = 0

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(AIFDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.encoder = EdgeLabelEncoder()
    
    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)
    
    @property
    def processed_file_names(self):
        return [file_name for file_name in os.listdir(self.processed_dir) if file_name.startswith("data_")]
    
    def download(self):
        pass

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.processed_file_names[idx]}'))
        return data

    def process(self):
        with cf.ThreadPoolExecutor() as executor:
            futures = []
            for raw_file in self.raw_file_names:
                file_path = os.path.join(self.raw_dir, raw_file)
                thread_utils.thread_safe_print(f"Processing {raw_file}")
                future = executor.submit(self._process_file, file_path)
                futures.append(future)
    
    def _process_file(self, file_path):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        processed_file_name = f'data_{file_name}.pt'
        processed_file_path = os.path.join(self.processed_dir, processed_file_name)

        with file_lock:
            if processed_file_name in os.listdir(self.processed_dir):
                thread_utils.thread_safe_print(f"Processed file {processed_file_name} already exists. Skipping...")
                return torch.load(processed_file_path)

        try:
            with open(file_path, "r") as json_file:
                json_data = json.load(json_file)

            graph = nx.Graph()
            for node in json_data["nodes"]:
                graph.add_node(node["nodeID"], type=node["type"], text=node["text"], embedding=None)
            for edge in json_data["edges"]:
                graph.add_edge(edge["fromID"], edge["toID"], type=None)

            aif_data = AIFData(graph=graph, name=file_name)

        except Exception as error:
            thread_utils.thread_safe_print(f"Failed to create graph {file_name}")
            return
        
        if self.pre_transform is not None:
            try:
                aif_data = self.pre_transform(aif_data)
            except Exception as transform_error:
                thread_utils.thread_safe_print(f"Failed to apply pre-transform for sub-graph {aif_data.name}: {str(transform_error)}")
                return
        
        subgraph_data = []

        # Split the graph into connected components
        connected_components = list(nx.connected_components(aif_data.graph))
        for i, component in enumerate(connected_components):
            subgraph = nx.Graph(aif_data.graph.subgraph(component))
            subgraph_name = f'{file_name}_component_{string.ascii_lowercase[i]}'
            component_data = AIFData(graph=subgraph, name=subgraph_name, num_nodes=subgraph.number_of_nodes())

            # Apply transformations
            component_data = self.edge_label_encoder(component_data)
            component_data = self.graph_to_pyg(component_data)

            # Filter components
            if self.pre_filter is not None and not self.pre_filter(component_data):
                thread_utils.thread_safe_print(f"Pre-filter rejected data in sub-graph {component_data.name}. Skipping this file.")
                continue
            
            subgraph_data.append(component_data)

        # Save the filtered components
        for component_data in subgraph_data:
            with file_lock:
                save_path = os.path.join(self.processed_dir, f'data_{component_data.name}.pt')
                torch.save(component_data, save_path)
                thread_utils.thread_safe_print(f"Saved {component_data.name}")
        
    def graph_to_pyg(self, data):
        try:
            nodes = {node: data.graph.nodes[node] for node in data.graph.nodes}
            edges = {edge: data.graph.edges[edge] for edge in data.graph.edges}

            node_mapping = {node: idx for idx, node in enumerate(nodes)}
            edge_index = [(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in edges]
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            node_embeddings = [nodes[node]['embedding'] for node in data.graph.nodes]
            node_embeddings_tensor = torch.tensor(node_embeddings, dtype=torch.float)
            node_embeddings_tensor = node_embeddings_tensor.view(node_embeddings_tensor.size(0), -1)

            centrality_values = [nodes[node].get('centrality', 0.0) for node in data.graph.nodes]
            centrality_tensor = torch.tensor(centrality_values, dtype=torch.float).unsqueeze(1)

            sentiment_values = [nodes[node].get('sentiment', 0.0) for node in data.graph.nodes]
            sentiment_tensor = torch.tensor(sentiment_values, dtype=torch.float).unsqueeze(1)

            node_features = torch.cat((node_embeddings_tensor, centrality_tensor, sentiment_tensor), dim=1)
            
            data.edge_index = edge_index
            data.y = data.edge_labels
            data.x = node_features

            thread_utils.thread_safe_print(f'Converted graph data to PyG data for {data.name}')
        
        except Exception as e:
            thread_utils.thread_safe_print(f'Failed to convert graph data to PyG data for {data.name}: {str(e)}')

        return data
    
    def edge_label_encoder(self, data):
        graph = data.graph

        try:
            unique_labels_before = set([graph.edges[src, dst].get("type") for src, dst in graph.edges()])
            thread_utils.thread_safe_print(f"Unique edge labels before encoding: {unique_labels_before}")

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

            thread_utils.thread_safe_print(f'Encoded edge labels for {data.name}')

        except Exception as e:
            thread_utils.thread_safe_print(f'Failed to encode edge labels for {data.name}: {str(e)}')

        return data