from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

class Doc2VecEmbedding:
    def __init__(self, model_path='path/to/doc2vec_model.bin'):
        self.model = Doc2Vec.load(model_path)

    def embed_text(self, text):
        tagged_data = [TaggedDocument(words=text.split(), tags=[0])]
        embeddings = self.model.infer_vector(tagged_data[0].words)
        return embeddings