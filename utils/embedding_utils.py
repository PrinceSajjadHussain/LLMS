# embedding_utils.py

from models.openai_model import get_openai_embedding
from models.llama_model import get_llama_embedding
from models.mistral_model import get_mistral_embedding

def get_embedding(text, model_name='openai'):
    if model_name == 'openai':
        return get_openai_embedding(text)
    elif model_name == 'llama':
        return get_llama_embedding(text)
    elif model_name == 'mistral':
        return get_mistral_embedding(text)
    else:
        raise ValueError("Unknown model name")
