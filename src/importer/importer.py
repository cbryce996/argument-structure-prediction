import json
import os
import threading
import concurrent.futures as cf
import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from importer.transforms import EdgeLabelEncoder
from utils import ThreadUtils

thread_utils = ThreadUtils()

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

        if processed_file_name in os.listdir(self.processed_dir):
            thread_utils.thread_safe_print(f"Processed file {processed_file_name} already exists. Skipping...")
            return torch.load(processed_file_path)
        
        try:
            with open(file_path, "r") as json_file:
                json_data = json.load(json_file)

            graph = nx.DiGraph()

            for node in json_data["nodes"]:
                graph.add_node(node["nodeID"], type=node["type"], text=node["text"], embedding=None)
            
            for edge in json_data["edges"]:
                graph.add_edge(edge["fromID"], edge["toID"], type=None)

            aif_data = AIFData(graph=graph, name=file_name)
        
        except Exception as error:
            thread_utils.thread_safe_print(f"Failed to create graph for {file_name}: {str(error)}")
            return

        if self.pre_transform is not None:
            try:
                aif_data = self.pre_transform(aif_data)
            except Exception as transform_error:
                thread_utils.thread_safe_print(f"Failed to apply pre-transform for {file_name}: {str(transform_error)}")
                return

        if self.pre_filter is not None and not self.pre_filter(aif_data):
            thread_utils.thread_safe_print(f"Pre-filter rejected data in file {file_name}. Skipping this file.")
            return

        try:
            with thread_utils.save_lock:
                torch.save(aif_data, processed_file_path)
                thread_utils.thread_safe_print(f"Saved {aif_data.name}")
        
        except Exception as save_error:
            thread_utils.thread_safe_print(f"Failed to save processed file for {file_name}: {str(save_error)}")
            return