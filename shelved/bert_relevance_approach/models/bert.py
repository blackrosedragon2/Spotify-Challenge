from transformers import BertTokenizer, BertForSequenceClassification


def load_model(model_name="bert-base-uncased"):
    """Load pretrained BertForSequenceClassification"""
    model = BertForSequenceClassification.from_pretrained(model_name)
    return model
