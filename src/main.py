import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data, DataLoader
from torch_geometric.transforms import Compose
from data import AIFDataset, RemoveLinkNodes, CreateBertEmbeddings, EdgeLabelEncoder, GraphToPyGData, EdgeLabelDecoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gnn import GCNClassifier
import time

if __name__ == "__main__":  
    start_time = time.time()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    encoder = EdgeLabelEncoder()
    decoder = EdgeLabelDecoder(label_encoder=encoder)

    transforms = Compose([RemoveLinkNodes(), EdgeLabelEncoder(), CreateBertEmbeddings(tokenizer, model), GraphToPyGData()])

    qt30_dataset = AIFDataset(root="/home/cameron/Dropbox/Uni/2024/CMP400/demo/data/QT30", pre_transform=transforms)

    us2016_dataset = AIFDataset(root="/home/cameron/Dropbox/Uni/2024/CMP400/demo/data/US2016", pre_transform=transforms)

    train_loader = DataLoader(qt30_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(us2016_dataset, batch_size=16, shuffle=True)

    # Assuming you have a DataLoader named 'train_loader' for training data
    model = GCNClassifier(input_dim=128*768, hidden_dim=256, output_dim=6)
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
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Test Accuracy: {accuracy * 100:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Total execution time: {elapsed_time:.2f} seconds')
