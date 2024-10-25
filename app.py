from flask import Flask, render_template, request, jsonify
import torch
import os
import warnings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='langchain')

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key='AIzaSyBwYQB_87zkmivHsYX9MR9PL8Hb3y77GYE')

def get_text(file_path):
    text = ""
    with open(file_path, 'r') as file:
        text = file.read()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    index_path = "faiss_index"
    try:
        vector_store.save_local(index_path)
        # print(f"FAISS index saved to {index_path}")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

def get_conversational_chain():
    prompt_template = """
    Answer the question as short as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, and if the context refers to daily routine or wishes, give the output; 
    otherwise, say "answer is not available in the context". Do not provide a wrong answer.

    Context:\n {context}\n
    Question: {question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_path = "faiss_index"
    try:
        new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        if docs:
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

            if "answer is not available in the context" in response["output_text"].lower():
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(user_question)
                print('rrr'+response.text)
                return response.text
            else:
                print(response["output_text"])
                # return response["output_text"]
                return response["output_text"]
        else:
            general_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
            general_response = general_model(user_question=user_question)
            print(general_response)
            return general_response

    except Exception as e:
        print(f"Error processing user input: {e}")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    raw_text = get_text('data3.txt')  # Assuming you are working with a text file.
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)  # Build FAISS vector store for similarity search.
    
    # Now process user input with FAISS and fallback to Generative AI.
    response = user_input(msg)
    return response  # Return response as JSON

if __name__ == '__main__':
    app.run()
