import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch.utils.data import random_split
from data import AIFDataset, KeepSelectedNodeTypes, RemoveLinkNodeTypes, CreateBertEmbeddings, EdgeLabelEncoder, GraphToPyGData, EdgeLabelDecoder, MinNodesAndEdges
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gnn_gcn import GCNClassifier
from gnn_sage import SAGEClassifier
import time

if __name__ == "__main__":
    start_time = time.time()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    encoder = EdgeLabelEncoder()
    decoder = EdgeLabelDecoder(label_encoder=encoder)

    transforms = Compose([
        KeepSelectedNodeTypes(types_to_keep=["I", "RA", "MA"]),
        RemoveLinkNodeTypes(types_to_remove=["RA", "MA"]),
        EdgeLabelEncoder(),
        CreateBertEmbeddings(tokenizer, model, 128),
        GraphToPyGData()
    ])
    filters = Compose([MinNodesAndEdges()])

    qt30_dataset = AIFDataset(root="/home/cameron/Dropbox/Uni/2024/CMP400/demo/data/QT30", pre_transform=transforms, pre_filter=filters)

    train_size = int(0.8 * len(qt30_dataset))
    val_size = int(0.1 * len(qt30_dataset))
    test_size = len(qt30_dataset) - train_size - val_size

    # Use random_split to create the datasets
    train_dataset, val_dataset, test_dataset = random_split(qt30_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = GCNClassifier(input_dim=128*768, hidden_dim=128, output_dim=2, num_fc_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    patience = 2  # Number of epochs to wait for improvement
    early_stop_counter = 0
    best_valid_loss = float('inf')  # Initialize with a large value

    for epoch in range(20):
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
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Test Accuracy: {accuracy * 100:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Total execution time: {elapsed_time:.2f} seconds')
