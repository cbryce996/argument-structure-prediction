import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data, DataLoader
from torch_geometric.transforms import Compose
from data import AIFDataset, RemoveLinkNodes, CreateBertEmbeddings, EdgeLabelEncoder, GraphToPyGData
from gnn import GCNClassifier

if __name__ == "__main__":    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    transforms = Compose([RemoveLinkNodes(), EdgeLabelEncoder(), CreateBertEmbeddings(tokenizer, model), GraphToPyGData()])

    dataset = AIFDataset(root="/home/cameron/Dropbox/Uni/2024/CMP400/demo/data/QT30", pre_transform=transforms)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print("Length of train_loader:", len(train_loader))

    # Assuming you have a DataLoader named 'train_loader' for training data
    model = GCNClassifier(input_dim=128*768, hidden_dim=256, output_dim=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Assuming you have a DataLoader named 'test_loader' for testing data
    for epoch in range(5):
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
        print(f'Epoch {epoch + 1}/{40}, Loss: {average_loss:.4f}')

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in train_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
