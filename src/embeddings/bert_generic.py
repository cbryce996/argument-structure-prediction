from transformers import AutoModel, AutoTokenizer


class BERTEmbedding:
    def __init__(
        self, model_name="bhadresh-savani/distilbert-base-uncased-sentiment-sst2"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings
