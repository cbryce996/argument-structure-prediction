import json
import os
import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import BaseTransform

class AIFData(Data):
    def __init__(self, x=None, edge_index=None, y=None, graph=None, name=None, **kwargs):
        super(AIFData, self).__init__(x=x, edge_index=edge_index, y=y, **kwargs)
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
    
    def from_nginx_graph(self):
        # Extract node features
            x = torch.tensor([self.graph.nodes[node]["embedding"] for node in self.graph.nodes], dtype=torch.float)

            # Create a mapping from edge labels (strings) to unique integer indices
            edge_label_to_index = {label: idx for idx, label in enumerate(set(self.graph.edges))}

            print()

            # Extract edge indices
            edge_index_list = [(edge_label_to_index[edge[0]], edge_label_to_index[edge[1]]) for edge in self.graph.edges]
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

            # Extract edge labels (if available)
            if "labels" in self.graph:
                y = torch.tensor(self.graph["labels"], dtype=torch.long)
                self.y = y

            # Update PyTorch Geometric Data attributes
            self.x = x
            self.edge_index = edge_index

class AIFDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(AIFDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)
    
    @property
    def processed_file_names(self):
        return ['processed_data.pt']
    
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

        data.from_nginx_graph()

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
        print(f'Removed link nodes from {data.name}')

        data.from_nginx_graph()

        return data
    
    def __repr__(self):
        return f"RemoveLinkNodes(link_node_types={self.link_node_types})"

class BinaryEdgeLabelEncoder(BaseTransform):
    def __init__(self):
        self.label_to_index = {'No-Relation': 0, 'Relation': 1}
        self.index_to_label = {0: 'No-Relation', 1: 'Relation'}
        self.num_labels = 2

    def __call__(self, data):
        graph = data.graph

        # Define the mapping for node types to edge labels
        type_to_label_mapping = {0: 'No-Relation', 1: 'Relation'}

        # Ensure that the graph has node attributes 'type' for each node
        for node in graph.nodes(data=True):
            if 'type' not in node[1]:
                print("Node is missing 'type' attribute.")
                return data

        # Iterate over all edges and set their labels based on the 'type' attribute of the source node
        for edge in graph.edges():
            source_type = graph.nodes[edge[0]]['type']
            label = type_to_label_mapping.get(source_type, type_to_label_mapping[0])  # Default to 'No-Relation'
            graph.edges[edge[0], edge[1]]['label'] = label

        data.from_nginx_graph()

        return data

    def __repr__(self):
        return f"BinaryEdgeLabelEncoder()"
    
class EdgeLabelEncoder(BaseTransform):
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
        for source, target in graph.edges():
            label = graph.edges[source, target].get("label")
            index = self.label_to_index[label]
            graph.edges[source, target]["label_encoded"] = index
            print(graph.edges[source, target]["label_encoded"])

        return data
    
    def __repr__(self):
        return f"EdgeLabelEncoder()"
    
class EdgeLabelDecoder(BaseTransform):
    def __init__(self, label_encoder):
        self.label_encoder = label_encoder

    def __call__(self, data):
        graph = data.graph

        # Decode edge label encodings in the graph
        for source, target in graph.edges():
            index = graph.edges[source, target].get("label_encoded")
            label = self.label_encoder.index_to_label.get(index)
            graph.edges[source, target]["label_decoded"] = label

        return data