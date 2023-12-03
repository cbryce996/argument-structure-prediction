import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch.utils.data import random_split, ConcatDataset
from data import AIFDataset, RemoveNodeTypes, RemoveLinkNodeTypes, CreateBertEmbeddings, EdgeLabelEncoder, GraphToPyGData, EdgeLabelDecoder, MinNodesAndEdges
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gnn_sage import SAGEClassifier
import networkx as nx
import time

if __name__ == "__main__":  
    start_time = time.time()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    encoder = EdgeLabelEncoder()
    decoder = EdgeLabelDecoder(label_encoder=encoder)

    transforms = Compose([RemoveNodeTypes(types_to_remove=["L"]), RemoveLinkNodeTypes(types_to_remove=["YA", "RA", "MA", "TA", "CA"]), EdgeLabelEncoder(), CreateBertEmbeddings(tokenizer, model, 128), GraphToPyGData()])
    filters = Compose([MinNodesAndEdges()])

    qt30_dataset = AIFDataset(root="/home/cameron/Dropbox/Uni/2024/CMP400/demo/data/QT30", pre_transform=transforms, pre_filter=filters)

    train_size = int(0.8 * len(qt30_dataset))
    test_size = int(0.1 * len(qt30_dataset))
    val_size = len(qt30_dataset) - train_size - test_size

    # Use random_split to create the datasets
    train_dataset, test_dataset, val_dataset = random_split(qt30_dataset, [train_size, test_size, val_size])

    # Create DataLoader for each set
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)  # No need to shuffle the test set
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)  # No need to shuffle the validation set

    # Assuming you have a DataLoader named 'train_loader' for training data
    model = SAGEClassifier(input_dim=128*768, hidden_dim=256, output_dim=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Assuming you have a DataLoader named 'test_loader' for testing data
    for epoch in range(20):
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
        print(f'Epoch {epoch + 1}/{20}, Loss: {average_loss:.4f}')

    # Evaluation
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
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Test Accuracy: {accuracy * 100:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Total execution time: {elapsed_time:.2f} seconds')