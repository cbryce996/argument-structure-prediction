from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose

from importer import AIFDataset
from importer.transforms import KeepSelectedNodeTypes, RemoveLinkNodeTypes, EdgeLabelEncoder, EmbedNodeText
from embeddings import Word2VecEmbedding, Doc2VecEmbedding, BagOfWordsEmbedding, BERTEmbedding
from models import GCNModel

transforms = Compose([
    KeepSelectedNodeTypes(types_to_keep=["I", "RA", "MA"]),
    RemoveLinkNodeTypes(types_to_remove=["RA", "MA"]),
    EdgeLabelEncoder(),
    EmbedNodeText(embedding_method='bert')
])

# Construct the PyTorch Geometric Dataset
aif_dataset = AIFDataset(root="./AIF", pre_transform=transforms)

# Define DataLoader
loader = DataLoader(aif_dataset, batch_size=1, shuffle=True)

# Initialize and train the GCN model
model = GCNModel(input_size=128, hidden_size=64, output_size=1)
model.train(loader, epochs=10)