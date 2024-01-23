from models import SequentialStackingModel
from training import train_model
from embeddings import Word2VecEmbedding, Doc2VecEmbedding, BagOfWordsEmbedding

# List of embedding classes
embedding_classes = [Word2VecEmbedding, Doc2VecEmbedding, BagOfWordsEmbedding]

# Directory to save trained models
save_dir = "models/trained_models/"

# Iterate over each embedding method
for embedding_class in embedding_classes:
    # Instantiate the Sequential Stacking model with the current embedding method
    model = SequentialStackingModel(embedding_class)

    # Train the model
    trained_model = train_model(model, dataset="your_dataset", epochs=your_epochs, batch_size=your_batch_size)

    # Save the trained model
    save_path = f"{save_dir}{embedding_class.__name__}_sequential_stacking_model.pth"
    trained_model.save_model(save_path)
