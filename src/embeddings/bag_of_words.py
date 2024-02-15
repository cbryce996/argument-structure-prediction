from sklearn.feature_extraction.text import CountVectorizer


class BagOfWordsEmbedding:
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def embed_text(self, text):
        embeddings = self.vectorizer.transform([text]).toarray()
        return embeddings
