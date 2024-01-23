from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

class Word2VecEmbedding:
    def __init__(self, model_name='word2vec-google-news-300'):
        self.model = Word2Vec.load(model_name)

    def embed_text(self, text):
        tokens = word_tokenize(text.lower())
        embeddings = [self.model.wv[word] for word in tokens if word in self.model.wv]
        return embeddings
