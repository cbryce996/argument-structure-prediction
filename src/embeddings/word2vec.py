from gensim.models import Word2Vec

class Word2VecEmbedding:
    def __init__(self, model_path='path/to/word2vec_model.bin'):
        self.model = Word2Vec.load(model_path)

    def embed_text(self, text):
        words = text.split()
        embeddings = [self.model[word] for word in words if word in self.model]
        return embeddings