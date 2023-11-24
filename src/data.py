import json
import os
import torch
import networkx as nx
from transformers import BertTokenizer, BertModel
from torch_geometric.transforms import Compose
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import BaseTransform

class AIFData(Data):
    def __init__(self, x=None, edge_index=None, y=None, attention_masks=None, graph=None, name=None):
        super(AIFData, self).__init__(x=x, edge_index=edge_index, y=y)
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
            print(f"Constructed a {self.graph}")
        
        except Exception as e:
            print(f"An error occurred while processing data: {str(e)}")
            self.graph = graph

class AIFDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(AIFDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.encoder = EdgeLabelEncoder()
        self.decoder = EdgeLabelDecoder(label_encoder=self.encoder)

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)
    
    @property
    def processed_file_names(self):
        return ['processed_data.pt']
    
    def access_encode(self):
        return self.encoder
    
    def access_decoder(self):
        return self.decoder
    
    def download(self):
        pass

    def process(self):
        data_list = []

        for raw_file in self.raw_file_names:
            if raw_file.endswith('.json'):
                json_file_path = os.path.join(self.raw_dir, raw_file)
                print (f"Processing {raw_file}")
                
                try:
                    with open(json_file_path, "r") as json_file:
                        graph_data = json.load(json_file)

                    if "nodes" not in graph_data or "edges" not in graph_data:
                        raise ValueError("JSON data must contain both 'nodes' and 'edges' keys.")

                    aif_data = AIFData(name=raw_file)
                    aif_data.process_json(graph_data)

                    data_list.append(aif_data)

                except KeyError:
                    print(f"KeyError in file {json_file_path}. Skipping this file.")
                except Exception as e:
                    print(f"Error processing file {json_file_path}: {str(e)}. Skipping this file.")

        # Check and apply the pre filter
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        # Check and apply the pre transforms
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

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
        data.x = torch.tensor(node_embeddings)

        # Convert attention masks
        node_masks = [nodes[node]['attention_mask'] for node in data.graph.nodes]
        data.attention_masks = torch.tensor(node_masks)

        # Convert edge index
        edge_index = [(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in edges]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        data.edge_index = edge_index

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

        for i, node in enumerate(graph.nodes()):
            graph.nodes[node]["embedding"] = embeddings[i].tolist()
            graph.nodes[node]["attention_mask"] = attention_mask[i].tolist()

        print(f'Created BERT embeddings for {data.name}')

        data.graph = graph

        return data

    def __repr__(self):
        return f"CreateBertEmbeddings()"

class RemoveLinkNodes(BaseTransform):
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

        except Exception as e:
            print(f"An error occurred while processing data: {str(e)}")

        print(f'Removed link nodes from {data.name}')
        
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
