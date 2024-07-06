from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain. llms import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
from dotenv import load_dotenv
import requests
import os

load_dotenv()

client = MongoClient(os.environ["MONGO_URI"])
dbName = "langchain_demo"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"


def generate_embedding(text: str) -> list[float]:
    response = requests.post(embedding_url, headers= {"Authorization": f"Bearer {os.environ["HF_TOKEN"]}"}, json= {"inputs": text})
    if(response.status_code != 200):
        raise ValueError(f"Request Failed with status code {response.status_code}: {response.text}")
    
    return response.json()


def load_data():
    loader = DirectoryLoader("./sample_files", glob="./*.txt", show_progress = True)
    data = loader.load()

    for doc in data:
        embedding = generate_embedding(doc.page_content)
        collection.insert_one({
            'metadata': doc.metadata,
            'page_content': doc.page_content,
            'embedding': embedding
        })
        print("Created new document")

# load_data()







# utils
