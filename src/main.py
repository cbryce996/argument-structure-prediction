import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from bert_embd import get_bert_embeddings
from data import AIFDataset, RemoveLinkNodes
from clf_gnn import GNNEdgeTypeClassifier

if __name__ == "__main__":
    
    dataset = AIFDataset(root="/home/cameron/Dropbox/Uni/2024/CMP400/demo/amf/data/US2016", transform=RemoveLinkNodes)

    print(dataset.len())

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GNNEdgeTypeClassifier()

    # Loss function for edge type classification (multi-label classification)
    criterion = nn.BCEWithLogitsLoss()

    # Example training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    #for epoch in range(100):
    #    model.train()
    #    total_loss = 0.0
#
    #    for data in data_loader:
    #        optimizer.zero_grad()
    #        out = model(data)
    #        loss = criterion(out, data.y)
    #        loss.backward()
    #        optimizer.step()
    #        total_loss += loss.item()
    #
    #    average_loss = total_loss / len(data_loader)
    #    print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}")