import os
import time
from collections import Counter

import networkx as nx
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from importer import AIFDataset
from importer.filters import IsConnected, MinNumberNodes, MinSparsity
from importer.transforms import (EmbedNodeText, ExtractAdditionalFeatures,
                                 KeepSelectedNodeTypes, RemoveLinkNodeTypes)
from models import GCNModel
from utils import PlotUtils as plt

# Define transforms
transforms = Compose(
    [
        KeepSelectedNodeTypes(),
        RemoveLinkNodeTypes(),
        EmbedNodeText(),
        ExtractAdditionalFeatures(),
    ]
)

filters = Compose([MinNumberNodes(), MinSparsity(), IsConnected()])

aif_dataset = AIFDataset(root="../data", pre_transform=transforms, pre_filter=filters)

train_size = int(0.7 * len(aif_dataset))
val_size = int(0.2 * len(aif_dataset))
test_size = len(aif_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    aif_dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)

class_distribution = Counter(
    [label for data in train_dataset for label in data.y.numpy()]
)
total_samples = sum(class_distribution.values())
class_weights = {
    label: total_samples / count for label, count in class_distribution.items()
}

weights = torch.tensor([class_weights[i] for i in range(len(class_distribution))])

min_lr = 0.0000000001
max_lr = 0.01
step_size_up = int(len(train_loader) * 2)

model = GCNModel(input_size=770, hidden_size=2048)
optimizer = torch.optim.SGD(model.parameters(), lr=min_lr, momentum=0.5)
criterion = nn.CrossEntropyLoss(weight=weights)

clr_scheduler = CyclicLR(
    optimizer,
    base_lr=min_lr,
    max_lr=max_lr,
    step_size_up=step_size_up,
    mode="triangular",
)

patience = 5
early_stop_counter = 0
best_valid_loss = float("inf")

train_losses = []
valid_losses = []

# Initialize lists to store training and validation accuracies
train_accuracies = []
valid_accuracies = []

# Training loop
start_time = time.time()
epochs = 20000
for epoch in range(epochs):
    # Training
    model.train()
    total_loss = 0
    total_samples = 0
    correct_train = 0

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += len(data.y)
        _, predicted = torch.max(output, 1)
        correct_train += (predicted == data.y).sum().item()

    average_loss = total_loss / len(train_loader.dataset)
    train_losses.append(average_loss)
    train_accuracy = correct_train / total_samples
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    total_valid_loss = 0
    correct_valid = 0

    with torch.no_grad():
        for data in valid_loader:
            output = model(data)
            loss = criterion(output, data.y)
            total_valid_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_valid += (predicted == data.y).sum().item()

    average_valid_loss = total_valid_loss / len(valid_loader.dataset)
    valid_losses.append(average_valid_loss)
    valid_accuracy = correct_valid / len(valid_loader.dataset)
    valid_accuracies.append(valid_accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs}, Training Loss: {average_loss:.4f}, Validation Loss: {average_valid_loss:.4f}"
    )

    clr_scheduler.step()

    if average_valid_loss < best_valid_loss:
        best_valid_loss = average_valid_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print(
            f"Early stopping after {epoch + 1} epochs. No improvement in validation loss."
        )
        break

plt.visualize_training_accuracy(
    train_accuracy=train_accuracies,
    valid_accuracy=valid_accuracies,
    output_dir="../results/plots/training",
)
plt.visualize_training_losses(
    train_losses=train_losses,
    valid_losses=valid_losses,
    output_dir="../results/plots/training",
)

# Test
model.eval()
all_labels = []
all_preds = []


def data_to_graph(data, labels):
    # Extract node features, edge indices, and edge labels from the data object
    x = data.x  # Node features
    edge_index = data.edge_index  # Edge indices
    edge_labels = labels  # Edge labels

    # Create a NetworkX graph
    graph = nx.Graph()

    # Add nodes with features to the graph
    num_nodes = x.size(0)
    for node_idx in range(num_nodes):
        graph.add_node(
            node_idx, features=x[node_idx].numpy()
        )  # Assuming x contains node features

    # Add edges with labels to the graph
    num_edges = edge_index.size(1)
    for edge_idx in range(num_edges):
        src, dst = edge_index[:, edge_idx]
        label = edge_labels[edge_idx].item()  # Assuming edge labels are scalar values
        graph.add_edge(src.item(), dst.item(), type=label)

    return graph


with torch.no_grad():
    for data in test_loader:
        output = model(data)
        predicted = torch.argmax(output, 1)
        all_labels.extend(data.y.cpu().numpy())

        all_preds.extend(predicted.cpu().numpy())

        graph = data_to_graph(data, data.y)
        graph_predicted = data_to_graph(data, predicted)

        plt.visualize_graphs(
            graph1=graph,
            graph2=graph_predicted,
            output_dir="../results/plots/evaluation",
        )

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(
    all_labels, all_preds, average="weighted", zero_division=True
)
recall = recall_score(all_labels, all_preds, average="weighted")
f1 = f1_score(all_labels, all_preds, average="weighted")
# roc_auc = roc_auc_score(all_labels, all_preds, average='macro')

print(
    f"Test Accuracy: {accuracy * 100:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")
