# llama_model.py

from transformers import AutoTokenizer, AutoModel
from config.settings import LLAMA_MODEL_NAME

llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
llama_model = AutoModel.from_pretrained(LLAMA_MODEL_NAME)

def get_llama_embedding(text):
    inputs = llama_tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = llama_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
