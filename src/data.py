import json
import os
import torch
import threading
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel
from torch_geometric.transforms import Compose
from torch_geometric.data import InMemoryDataset, Data, Batch
from torch_geometric.transforms import BaseTransform

print_lock = threading.Lock()
save_lock = threading.Lock()

def thread_safe_print(message):
    with print_lock:
        print(message)

class AIFData(Data):
    def __init__(self, x=None, y=None, attention_masks=None, graph=None, name=None):
        super(AIFData, self).__init__(x=x, y=y)
        self.graph = graph
        self.name = name

    def process_json(self, data):
        graph = nx.DiGraph()

        try:
            for i, node in enumerate(data["nodes"]):
                graph.add_node(node["nodeID"], type=node["type"], text=node["text"], embedding=[])
            
            for edge in data["edges"]:
                graph.add_edge(edge["fromID"], edge["toID"], type=[])
    
            self.graph = graph
            thread_safe_print(f"Constructed a {self.graph}")
        
        except Exception as e:
            thread_safe_print(f"An error occurred while processing data: {str(e)}")
            self.graph = graph

class AIFDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, encoder=None, decoder=None):
        super(AIFDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.encoder = encoder if encoder is not None else EdgeLabelEncoder()
        self.decoder = decoder if decoder is not None else EdgeLabelDecoder(label_encoder=self.encoder)
        self.processed_data = None

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return [f'{processed_name}.pt' for processed_name in self.processed_names]

    @property
    def processed_names(self):
        processed_files = os.listdir(self.processed_dir)
        return [filename.replace(".pt", "") for filename in processed_files]

    def access_encode(self):
        return self.encoder

    def access_decoder(self):
        return self.decoder

    def download(self):
        pass

    def process(self):
        with ThreadPoolExecutor() as executor:
            futures = []
            for idx, raw_file in enumerate(self.raw_file_names):
                if raw_file.endswith('.json'):
                    json_file_path = os.path.join(self.raw_dir, raw_file)
                    thread_safe_print(f"Processing {raw_file}")
                    future = executor.submit(self._process_single_file, json_file_path)
                    futures.append(future)     

    def _process_single_file(self, json_file_path):
        processed_name = os.path.splitext(os.path.basename(json_file_path))[0]

        # Check if the processed file already exists
        processed_file_path = os.path.join(self.processed_dir, f'data_{processed_name}.pt')
        if os.path.exists(processed_file_path):
            thread_safe_print(f"Processed file {processed_name} already exists. Skipping...")
            return torch.load(processed_file_path)

        try:
            with open(json_file_path, "r") as json_file:
                graph_data = json.load(json_file)

            if "nodes" not in graph_data or "edges" not in graph_data:
                raise ValueError("JSON data must contain both 'nodes' and 'edges' keys.")

            aif_data = AIFData(name=processed_name)
            aif_data.process_json(graph_data)

            # Apply pre-filter
            if self.pre_filter is not None and not self.pre_filter(aif_data):
                return None

            # Apply pre-transform
            if self.pre_transform is not None:
                aif_data = self.pre_transform(aif_data)

            with save_lock:
                thread_safe_print(f"Saving {aif_data.name}")
                torch.save(aif_data, os.path.join(self.processed_dir, f'data_{aif_data.name}.pt'))

        except KeyError:
            thread_safe_print(f"KeyError in file {processed_name}. Skipping this file.")

        except Exception as e:
            thread_safe_print(f"Error processing file {processed_name}: {str(e)}. Skipping this file.")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.processed_file_names[idx]}'))
        return data

class GraphToPyGData(BaseTransform):
    def __init__(self):
        super(GraphToPyGData, self).__init__()

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

class CreateBertEmbeddings(BaseTransform):
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

        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)

        for i, node in enumerate(graph.nodes()):
            graph.nodes[node]["embedding"] = masked_embeddings[i].tolist()

        thread_safe_print(f'Created BERT embeddings for {data.name}')

        data.graph = graph

        return data

    def __repr__(self):
        return f"CreateBertEmbeddings()"

class ProcessGraphData(BaseTransform):
    def __init__(self, link_node_types=None):
        self.link_node_types = link_node_types or ["YA", "RA", "MA", "TA", "CA"]

    def __call__(self, data):
        graph = data.graph

        try:
            link_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get("type") in self.link_node_types]

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

            locution_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get("type") == "L"]
            
            for node in locution_nodes:
                graph.remove_node(node)

        except Exception as e:
            thread_safe_print(f"An error occurred while processing the graph data: {str(e)}")

        thread_safe_print(f'Successfully processed graph data from {data.name}')
        
        data.graph = graph

        return data
    
    def __repr__(self):
        return f"RemoveLinkNodes(link_node_types={self.link_node_types})"

class EdgeLabelEncoder(BaseTransform):
    def __init__(self):
        self.label_to_index = {}
        self.index_to_label = {}
        self.num_labels = 0

    def __call__(self, data):
        graph = data.graph

        unique_labels = set([graph.edges[src, dst].get("type") for src, dst in graph.edges()])
        
        for label in unique_labels:
            if label not in self.label_to_index:
                self.label_to_index[label] = self.num_labels
                self.index_to_label[self.num_labels] = label
                self.num_labels += 1

        label_index = [self.label_to_index[graph.edges[src, dst].get("type")] for src, dst in graph.edges()]
        label_index = torch.tensor(label_index, dtype=torch.long)

        data.edge_labels = label_index

        thread_safe_print(f'Encoded edge labels for {data.name}')

        return data
    
    def __repr__(self):
        return f"EdgeLabelEncoder()"
    
class EdgeLabelDecoder(BaseTransform):
    def __init__(self, label_encoder):
        self.label_encoder = label_encoder

    def __call__(self, data):
        # Decode edge label indices in the PyTorch Geometric Data object
        decoded_labels = [self.label_encoder.index_to_label[idx.item()] for idx in data.edge_labels]

        # Store the decoded edge labels in the PyTorch Geometric Data object
        data.edge_labels_decoded = decoded_labels

        return data
