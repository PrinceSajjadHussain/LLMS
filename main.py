# main.py

from utils.response_generator import generate_response

if __name__ == "__main__":
    user_query = "I forgot my password, what should I do?"
    response = generate_response(user_query)
    print(response)
