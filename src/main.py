import time
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from importer import AIFDataset
from importer.transforms import KeepSelectedNodeTypes, RemoveLinkNodeTypes, EdgeLabelEncoder, EmbedNodeText, GraphToPyG, KeepLargestConnectedComponent
from importer.filters import MinNodesAndEdges, MinSparsityAndConnectivity
from embeddings import Word2VecEmbedding, Doc2VecEmbedding, BagOfWordsEmbedding, BERTEmbedding
from models import GCNModel, GATModel, SequentialStackingModel
import torch
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
import os

def visualize_graph(data, output_dir):
    graph = data.graph

    plt.figure(figsize=(10, 10))

    # Adjust the layout to spread out nodes and prevent label overlap
    pos = nx.spring_layout(graph, seed=42, k=0.2)  # Adjust the value of k as needed

    # Draw nodes as squares with text within
    node_labels = {node: graph.nodes[node]['text'] for node in graph.nodes}
    nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=200)

    # Draw edges with edge labels
    nx.draw_networkx_edges(graph, pos, edge_color='gray', width=0.5)
    edge_labels = {(edge[0], edge[1]): graph.edges[edge]['type'] for edge in graph.edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f'Graph Visualization: {data.name}')
    plt.axis('off')  # Hide axis
    plt.tight_layout()  # Adjust layout to prevent labels from being cut off
    plt.savefig(os.path.join(output_dir, f'{data.name}_graph.png'))  # Save the plot
    plt.close()

# Define transforms
transforms = Compose([
    KeepSelectedNodeTypes(types_to_keep=["I", "RA", "MA"]),
    RemoveLinkNodeTypes(types_to_remove=["RA", "MA"]),
    EmbedNodeText(embedding_method='bert-generic'),
    KeepLargestConnectedComponent(),
    EdgeLabelEncoder(),
    GraphToPyG()
])

# Define filters
filters = Compose([MinNodesAndEdges(), MinSparsityAndConnectivity()])

# Construct the PyTorch Geometric Dataset
qt30_dataset = AIFDataset(root="../data/qt30", pre_transform=transforms, pre_filter=filters)

for data in qt30_dataset:
    visualize_graph(data, '../results/plots/graphs')

# Split the dataset
train_size = int(0.6 * len(qt30_dataset))
val_size = int(0.3 * len(qt30_dataset))
test_size = len(qt30_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(qt30_dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# Initialize model, optimizer, and criterion
model = GCNModel(input_size=3072, hidden_size=256)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

patience = 20
early_stop_counter = 0
best_valid_loss = float('inf') 

# Training loop
start_time = time.time()
for epoch in range(200):
    # Training
    model.train()
    total_loss = 0
    total_samples = 0

    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += len(data.y)

    average_loss = total_loss / len(train_loader.dataset)
    print(f'Epoch {epoch + 1}/{20}, Training Loss: {average_loss:.4f}')

    # Validation
    model.eval()
    total_valid_loss = 0

    with torch.no_grad():
        for data in valid_loader:
            output = model(data)
            loss = criterion(output, data.y)
            total_valid_loss += loss.item()

    average_valid_loss = total_valid_loss / len(valid_loader.dataset)
    print(f'Epoch {epoch + 1}/{20}, Validation Loss: {average_valid_loss:.4f}')

    if average_valid_loss < best_valid_loss:
        best_valid_loss = average_valid_loss
        early_stop_counter = 0  # Reset the counter if there's improvement
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print(f'Early stopping after {epoch + 1} epochs. No improvement in validation loss.')
        break

# Test
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for data in test_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        all_labels.extend(data.y.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Calculate and print various metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=True)
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f'Test Accuracy: {accuracy * 100:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

end_time = time.time()
elapsed_time = end_time - start_time
print(f'Total execution time: {elapsed_time:.2f} seconds')