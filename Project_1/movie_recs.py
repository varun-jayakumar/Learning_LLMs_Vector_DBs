import pymongo
import requests
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

# connecting to DB
client = pymongo.MongoClient(os.environ["MONGO_URI"])
db = client.sample_mflix
collection = db.movies

#  Embeddings logic
hf_token = os.environ["HF_TOKEN"]
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def generate_embedding(text: str) -> list[float]:
    response = requests.post(embedding_url, headers= {"Authorization": f"Bearer {hf_token}"}, json= {"inputs": text})
    if(response.status_code != 200):
        raise ValueError(f"Request Failed with status code {response.status_code}: {response.text}")
    
    return response.json()

# Generate Embeddings for movies (50 entries)
def update_documents_in_db():
    print("starting Update")
    for doc in collection.find({'plot': {"$exists": True}}).limit(50):
        doc['plot_embeddings_hf'] = generate_embedding(doc['plot'])
        collection.replace_one({'_id': doc['_id']}, doc)

    print("Update Complete")

# update_documents_in_db()



# Semantic Search demonstraction
query = "going thoriugh tough times"

results = collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": generate_embedding(query),
            "path": "plot_embeddings_hf",
            "numCandidates": 100,
            "limit": 4,
            "index": "PlotSematicSearch"
        }
    }
])

for result in results:
    print(result['plot'])


