import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data, DataLoader
from torch_geometric.transforms import Compose
from data import AIFDataset, RemoveLinkNodes, CreateBertEmbeddings, BinaryEdgeLabelEncoder
from gnn import GNNClassifier

if __name__ == "__main__":

    # Instantiate tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    # Define the transforms to be applied to the dataset
    transforms = Compose([CreateBertEmbeddings(tokenizer, model), RemoveLinkNodes(), BinaryEdgeLabelEncoder()])
    
    dataset = AIFDataset(
        root="/home/cameron/Dropbox/Uni/2024/CMP400/demo/data/QT30",
        pre_transform=transforms
    )

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GNNClassifier(in_channels=768, hidden_channels=64, num_classes=2)

    # Loss function for edge type classification (multi-label classification)
    criterion = nn.CrossEntropyLoss()

    # Example training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data in data_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)  # Assuming 'x' contains node features
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # For accuracy calculation
            predictions = out.argmax(dim=1)
            correct = (predictions == data.y).sum().item()
            total_correct += correct
            total_samples += data.num_graphs

        average_loss = total_loss / len(data_loader)
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
