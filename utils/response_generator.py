# response_generator.py

from models.openai_model import get_openai_completion
from vector_db.pinecone_client import query_pinecone
from utils.embedding_utils import get_embedding

def generate_response(user_query):
    matches = query_pinecone(user_query, embedding_function=get_embedding)
    if matches:
        best_match_id = matches[0]['id']
        best_match_text = next(item['text'] for item in data if item['id'] == best_match_id)
        response = get_openai_completion(f"User asked: {user_query}. Based on this FAQ: {best_match_text}, provide a detailed response.")
        return response
    else:
        return "Sorry, I couldn't find an answer to your question."
