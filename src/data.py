import json
import os
import threading
import torch
import networkx as nx
import concurrent.futures as cf
from transformers import BertTokenizer, BertModel
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import BaseTransform

print_lock = threading.Lock()
save_lock = threading.Lock()

def thread_safe_print(message):
    with print_lock:
        print(message)

class AIFData(Data):
    def __init__(self, x=None, y=None, graph=None, name=None):
        super(AIFData, self).__init__(x=x, y=y)
        self.graph = graph
        self.name = name

class AIFDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(AIFDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.encoder = EdgeLabelEncoder()
        self.decoder = EdgeLabelDecoder(label_encoder=self.encoder)
    
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

    # Starts threads for processing of files
    def process(self):
        with cf.ThreadPoolExecutor() as executor:
            futures = []
            for raw_file in self.raw_file_names:
                file_path = os.path.join(self.raw_dir, raw_file)
                thread_safe_print(f"Processing {raw_file}")
                future = executor.submit(self._process_file, file_path)
                futures.append(future)
            
            #cf.wait(futures)
    
    # Processes a single file
    def _process_file(self, file_path):
        # Get name and file path
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        processed_file_name = f'data_{file_name}.pt'  # Adjust the slicing based on your specific file name pattern
        processed_file_path = os.path.join(self.processed_dir, processed_file_name)

        # If processed file already exists then load
        if processed_file_name in os.listdir(self.processed_dir):
            thread_safe_print(f"Processed file {processed_file_name} already exists. Skipping...")
            return torch.load(processed_file_path)
        
        #! Implement AIF format validator
        # Import json
        try:
            with open(file_path, "r") as json_file:
                json_data = json.load(json_file)

        except Exception as error:
            thread_safe_print(f"Failed to open file {file_name}: {str(error)}")
            return

        # Convert json to graph
        try:
            graph = nx.DiGraph()

            for node in json_data["nodes"]:
                graph.add_node(node["nodeID"], type=node["type"], text=node["text"], embedding=None)
            
            for edge in json_data["edges"]:
                graph.add_edge(edge["fromID"], edge["toID"], type=None)

            aif_data = AIFData(graph=graph, name=file_name)
        
        except Exception as error:
            thread_safe_print(f"Failed to create graph for {file_name}: {str(error)}")
            return

        # Run pre_transform
        if self.pre_transform is not None:
            aif_data = self.pre_transform(aif_data)

        # Run pre_filter
        if self.pre_filter is not None and not self.pre_filter(aif_data):
             thread_safe_print(f"Pre-filter rejected data in file {file_name}. Skipping this file.")
             return

        # Save with lock
        try:
            with save_lock:
                torch.save(aif_data, processed_file_path)
                thread_safe_print(f"Saved {aif_data.name}")
        
        except Exception as error:
            thread_safe_print(f"Failed to save processed file for {file_name}: {str(error)}")
            return
        
class GraphToPyGData(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data):
        # Get graph data
        nodes = {node: data.graph.nodes[node] for node in data.graph.nodes}
        edges = {edge: data.graph.edges[edge] for edge in data.graph.edges}

        # Create node mapping
        node_mapping = {node: idx for idx, node in enumerate(nodes)}

        # Convert bert embeddings
        node_embeddings = [nodes[node]['embedding'] for node in data.graph.nodes]
        node_embeddings_tensor = torch.tensor([embeddings for embeddings in node_embeddings])
        flattened_embeddings = node_embeddings_tensor.view(node_embeddings_tensor.size(0), -1)
        data.x = flattened_embeddings

        # Convert edge index
        edge_index = [(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in edges]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        data.edge_index = edge_index

        # Convert edge labels
        data.y = data.edge_labels

        thread_safe_print(f'Converted graph data to PyG data for {data.name}')

        return data

#! Typed to "I" nodes for now
# Sets the minimum node and edge value filter
class MinNodesAndEdges(BaseTransform):
    def __init__(self, min_nodes=3, min_edges=3, min_ma_edges=3, min_ra_edges=3):
        self.min_nodes = min_nodes
        self.min_edges = min_edges
        self.min_ma_edges = min_ma_edges
        self.min_ra_edges = min_ra_edges

    def __call__(self, data):
        num_i_nodes = sum(1 for node, attrs in data.graph.nodes(data=True) if attrs.get("type") == "I")
        num_ma_edges = sum(1 for _, _, attrs in data.graph.edges(data=True) if attrs.get("type") == "MA")
        num_ra_edges = sum(1 for _, _, attrs in data.graph.edges(data=True) if attrs.get("type") == "RA")
        num_edges = data.graph.number_of_edges()

        if getattr(data, 'num_unique_labels', 0) > 2:
            thread_safe_print(f"Rejected data {data.name} due to more than 2 unique edge labels.")
            return False

        return (
            num_i_nodes >= self.min_nodes
            and num_edges >= self.min_edges
            and num_ma_edges >= self.min_ma_edges
            and num_ra_edges >= self.min_ra_edges
        )


# Keep only specified node types and remove all others
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

            thread_safe_print(f"Successfully kept only node types {self.types_to_keep} in {data.name}")

            return data
            
        except Exception as error:
            thread_safe_print(f"Failed to keep only node types {self.types_to_keep} in {data.name}: {str(error)}")

# Removes node types from graph maintains edge connections
class RemoveLinkNodeTypes(BaseTransform):
    def __init__(self, types_to_remove):
        self.types_to_remove = types_to_remove

    def __call__(self, data):
        graph = data.graph

        try:
            link_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get("type") in self.types_to_remove]

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

            data.graph = graph

            thread_safe_print(f"Successfully removed link node types {self.types_to_remove} in {data.name}")

            return data

        except Exception as error:
            thread_safe_print(f"Failed to remove link node types {self.types_to_remove} in {data.name}: {str(error)}")

class CreateBertEmbeddings(BaseTransform):
    def __init__(self, tokenizer=None, model=None, max_length=128):
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length

    def __call__(self, data):
        graph = data.graph

        node_text = [graph.nodes[node]["text"] for node in graph.nodes()]

        tokenized_inputs = self.tokenizer(
            node_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        attention_mask = tokenized_inputs["attention_mask"]

        with torch.no_grad():
            output = self.model(**tokenized_inputs)
            embeddings = output.last_hidden_state

        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)

        for i, node in enumerate(graph.nodes()):
            graph.nodes[node]["embedding"] = masked_embeddings[i].tolist()

        thread_safe_print(f"Created BERT embeddings for {data.name}")

        data.graph = graph

        return data

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

class EdgeLabelDecoder(BaseTransform):
    def __init__(self, label_encoder):
        self.label_encoder = label_encoder

    def __call__(self, data):
        # Decode edge label indices in the PyTorch Geometric Data object
        decoded_labels = [self.label_encoder.index_to_label[idx.item()] for idx in data.edge_labels]

        # Store the decoded edge labels in the PyTorch Geometric Data object
        data.edge_labels_decoded = decoded_labels

        return data