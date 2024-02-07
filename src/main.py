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
from torch.optim.lr_scheduler import StepLR
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

def visualize_training_losses(train_losses, valid_losses, output_dir):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-')
    plt.plot(epochs, valid_losses, label='Validation Loss', marker='o', linestyle='-')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_validation_losses.png'))
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

#for data in qt30_dataset:
#    visualize_graph(data, '../results/plots/graphs')

# Split the dataset
train_size = int(0.6 * len(qt30_dataset))
val_size = int(0.2 * len(qt30_dataset))
test_size = len(qt30_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(qt30_dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

# Initialize model, optimizer, and criterion
model = GCNModel(input_size=768, hidden_size=5048)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.9)
criterion = nn.CrossEntropyLoss()

patience = 5
early_stop_counter = 0
best_valid_loss = float('inf')

train_losses = []
valid_losses = []

# Training loop
start_time = time.time()
epochs = 500
for epoch in range(epochs):
    # Training
    model.train()
    total_loss = 0
    total_samples = 0

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += len(data.y)

    average_loss = total_loss / len(train_loader.dataset)
    train_losses.append(average_loss)

    # Validation
    model.eval()
    total_valid_loss = 0

    with torch.no_grad():
        for data in valid_loader:
            output = model(data)
            loss = criterion(output, data.y)
            total_valid_loss += loss.item()

    average_valid_loss = total_valid_loss / len(valid_loader.dataset)
    valid_losses.append(average_valid_loss)

    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {average_loss:.4f}, Validation Loss: {average_valid_loss:.4f}')

visualize_training_losses(train_losses, valid_losses, '../results/plots')

# Test
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for data in test_loader:
        output = model(data)
        predicted = torch.argmax(output, 1)
        all_labels.extend(data.y.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Calculate and print various metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f'Test Accuracy: {accuracy * 100:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

end_time = time.time()
elapsed_time = end_time - start_time
print(f'Total execution time: {elapsed_time:.2f} seconds')