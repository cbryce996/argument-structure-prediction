from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize


class Doc2VecEmbedding:
    def __init__(self, model_name="doc2vec-wiki-english"):
        self.model = Doc2Vec.load(model_name)

    def embed_text(self, text):
        tokens = word_tokenize(text.lower())
        doc = TaggedDocument(words=tokens, tags=[0])
        return self.model.infer_vector(doc.words)
