# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # Load the dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
# data = pd.read_csv(url, header=None, names=columns)

# # Preprocessing
# X = data.drop('class', axis=1)
# y = data['class']

# # Normalize features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Save preprocessed data
# pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
# pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
# pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
# pd.DataFrame(y_test).to_csv('y_test.csv', index=False)


# # Model Training
# # class

# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import joblib

# # Load preprocessed data
# X_train = pd.read_csv('X_train.csv')
# y_train = pd.read_csv('y_train.csv').values.ravel()

# # Train the model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Save the model
# joblib.dump(model, 'model.pkl')

# # Evaluate the model
# X_test = pd.read_csv('X_test.csv')
# y_test = pd.read_csv('y_test.csv').values.ravel()
# y_pred = model.predict(X_test)
# print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")












# import pandas as pd
# from sklearn.model_selection import train_test_split

# #load the dataset
# url=""
# columns=['','','','','']
# data=pd.read_csv(url, header=None , names=columns)

# #Preprocessing
# X=data.drop('class',axis=1)
# y= data['class']

# #Normalize features
# scaler= StandardScaler()
# x_scaled=scaler.fit_transform(X)

# #Split the dataset
# X_train, X_test, y_train, y_test=train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
# pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
# pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
# pd.DataFrame(y_test).to_csv('y_test.csv', index=False)



# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import joblib

# #Load preprocessed data
# X_train= pd.read_csv('X_train.csv')
# y_train= pd.read_csv('y_train.csv').values_ravel()
# #train the model
# model=RandomForestClassifier(e_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# #Save the model 
# joblib.dump(model, 'model.pkl')

# #Evaluate the model
# s
# X_test= pd.read_csv('X_test.csv')
# y_test= pd.read_csv('y_test.csv').values.ravels()
# y_pred= model.predict(X_test)
# print(f"Model Accuracy: "{accuracy_score(y_test, y_pred)})



import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import pytesseract
from pdf2image import convert_from_path
import logging

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Setup logging
logging.basicConfig(level=logging.INFO)

def extract_text_from_image(pdf_path):
    """Use OCR to extract text from PDF images."""
    images = convert_from_path(pdf_path)
    text = ''
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs, supporting both text-based and scanned PDFs."""
    text = ""
    with ThreadPoolExecutor() as executor:
        results = executor.map(extract_single_pdf_text, pdf_docs)
    for result in results:
        text += result
    return text

def extract_single_pdf_text(pdf):
    try:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                logging.info(f"Page requires OCR extraction: {pdf.name}")
                text += extract_text_from_image(pdf)
        return text
    except Exception as e:
        logging.error(f"Error processing PDF {pdf.name}: {e}")
        return ""

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,  # Reduced chunk size for better handling
        chunk_overlap=500  # Adjusted overlap for more accurate context matching
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory='chroma_index')
    return vector_store

def get_conversational_chain():
    prompt_template = """
    You have been provided with a PDF document that contains multiple pages, each with an invoice number at the top. 
    Your task is to find the page that corresponds to the provided invoice number and retrieve all the details from that specific page. 
    If the invoice number is not found in the context provided, respond with "Invoice number not found in the document."
    Ensure that you only provide information from the page that matches the given invoice number.

    Invoice Number: {invoice_number}

    Context:
    {context}

    Details for Invoice Number {invoice_number}:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "invoice_number"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, vector_store):
    # Perform the similarity search to find the relevant document chunk
    docs = vector_store.similarity_search(user_question, k=5)  # Increase k for better search results 
    # Combine the chunks into one context string
    context = "\n".join([doc.page_content for doc in docs])
    
    # Generate the conversational response
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "context": context, "invoice_number": user_question},
        return_only_outputs=True
    )
    
    return response["output_text"]

def main():
    st.title("Advanced PDF Chatbot")
    st.write("Upload PDFs and query specific invoice details efficiently with AI.")

    pdf_docs = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
    
    if pdf_docs:
        st.write("Processing the PDF(s)...")
        with st.spinner("Extracting text..."):
            text = get_pdf_text(pdf_docs)
        st.success("PDF text extraction complete!")
        
        text_chunks = get_text_chunks(text)
        vector_store = get_vector_store(text_chunks)
        st.success("PDF processed and indexed successfully!")

        user_question = st.text_input("Enter the Invoice Number to get details:")
        
        if user_question:
            st.write("Generating response...")
            response = user_input(user_question, vector_store)
            st.markdown("### Reply: ")
            st.write(response)

if __name__ == '__main__':
    main()






