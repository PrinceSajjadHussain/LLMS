# openai_model.py

import openai
from config.settings import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def get_openai_completion(prompt, model='gpt-4'):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def get_openai_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']
