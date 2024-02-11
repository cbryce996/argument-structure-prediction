import torch
from transformers import AutoTokenizer, AutoModel

class BERTEmbedding:
    def __init__(self, model_name='ali2066/distilbert-base-uncased-finetuned-sst-2-english-finetuned-argumentative'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings