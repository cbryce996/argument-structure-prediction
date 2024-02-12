from collections import Counter
import time
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from importer import AIFDataset
from importer.transforms import KeepSelectedNodeTypes, RemoveLinkNodeTypes, EdgeLabelEncoder, EmbedNodeText, GraphToPyG, KeepLargestConnectedComponent, ExtractAdditionalFeatures
from importer.filters import MinNumberNodes, MinSparsityAndConnectivity
from embeddings import Word2VecEmbedding, Doc2VecEmbedding, BagOfWordsEmbedding, BERTEmbedding
from models import GCNModel, GATModel, SequentialStackingModel
import torch
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CyclicLR
import matplotlib.pyplot as plt
from collections import Counter
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
    KeepSelectedNodeTypes(),
    RemoveLinkNodeTypes(),
    EmbedNodeText(),
    ExtractAdditionalFeatures()
])

filters = Compose([MinNumberNodes(), MinSparsityAndConnectivity()])

aif_dataset = AIFDataset(root="../data", pre_transform=transforms, pre_filter=filters)

for i in range(4):
    visualize_graph(aif_dataset[i], f"../results/plots/graphs")

train_size = int(0.7 * len(aif_dataset))
val_size = int(0.2 * len(aif_dataset))
test_size = len(aif_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(aif_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

class_distribution = Counter([label for data in train_dataset for label in data.y.numpy()])
total_samples = sum(class_distribution.values())
class_weights = {label: total_samples / count for label, count in class_distribution.items()}

weights = torch.tensor([class_weights[i] for i in range(len(class_distribution))])

min_lr = 0.00000001
max_lr = 0.001
step_size_up = int(len(train_loader) * 2)

model = GCNModel(input_size=770, hidden_size=1024)
optimizer = torch.optim.SGD(model.parameters(), lr=min_lr, momentum=0.5)
criterion = nn.CrossEntropyLoss(weight=weights)

clr_scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=step_size_up, mode='triangular'
)

patience = 5
early_stop_counter = 0
best_valid_loss = float('inf')

train_losses = []
valid_losses = []

# Training loop
start_time = time.time()
epochs = 20000
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

    clr_scheduler.step()

    if average_valid_loss < best_valid_loss:
        best_valid_loss = average_valid_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print(f'Early stopping after {epoch + 1} epochs. No improvement in validation loss.')
        break

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
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=True)
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f'Test Accuracy: {accuracy * 100:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

end_time = time.time()
elapsed_time = end_time - start_time
print(f'Total execution time: {elapsed_time:.2f} seconds')