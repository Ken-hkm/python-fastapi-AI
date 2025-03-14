import argparse
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--collection", required=True, help="MongoDB collection name")
args = parser.parse_args()

# MongoDB connection
MONGODB_URI = os.environ.get("MONGODB_URI")
client = MongoClient(MONGODB_URI, tls=True, tlsAllowInvalidCertificates=True)
db = client["Resume"]
collection = db[args.collection]

# Load the model for embeddings
embedding_model = SentenceTransformer('all-MPnet-base-v2')

# Define fields to ignore
ignore_fields = {"_id", "embedding"}

# Extract documents and generate embeddings
documents_to_update = collection.find({})

for doc in documents_to_update:
    text = " ".join(str(value) for key, value in doc.items() if key not in ignore_fields)

    # Generate embedding and convert to list
    embedding = embedding_model.encode(text).tolist()

    # Update the document with the generated embedding
    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"embeddings": embedding}}
    )

print(f"Embeddings have been added to all documents in the '{args.collection}' collection.")