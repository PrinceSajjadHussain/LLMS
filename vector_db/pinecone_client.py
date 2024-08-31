# pinecone_client.py

import os
from pinecone import Pinecone, ServerlessSpec
from config.settings import PINECONE_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'faq-index'

def create_index(dimension):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-west-2'
            )
        )

def get_index():
    return pc.get_index(index_name)

def index_data(data, embedding_function):
    index = get_index()
    vectors = [(item['id'], embedding_function(item['text'])) for item in data]
    index.upsert(vectors)

def query_pinecone(query_text, embedding_function):
    index = get_index()
    query_vector = embedding_function(query_text)
    result = index.query(query_vector, top_k=5)
    return result['matches']
