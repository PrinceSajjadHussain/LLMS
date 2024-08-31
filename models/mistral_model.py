# mistral_model.py

from transformers import AutoTokenizer, AutoModel
from config.settings import MISTRAL_MODEL_NAME

mistral_tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL_NAME)
mistral_model = AutoModel.from_pretrained(MISTRAL_MODEL_NAME)

def get_mistral_embedding(text):
    inputs = mistral_tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = mistral_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
