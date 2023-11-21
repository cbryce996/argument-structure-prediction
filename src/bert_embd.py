from transformers import BertTokenizer, BertModel
import torch


def get_bert_embeddings(texts, max_seq_length=128):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    tokenized_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_length
    )

    attention_mask = tokenized_inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(**tokenized_inputs)
        embeddings = outputs.last_hidden_state

    return embeddings, attention_mask