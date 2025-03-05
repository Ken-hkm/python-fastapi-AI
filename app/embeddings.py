from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# MongoDB connection
MONGODB_URI = os.environ.get("MONGODB_URI")
client = MongoClient(MONGODB_URI, tls=True, tlsAllowInvalidCertificates=True)
db = client["Resume"]
collection = db["personal-info"]

# Load the model for embeddings
embedding_model = SentenceTransformer('all-MPnet-base-v2')

# Define fields to ignore
ignore_fields = {"_id", "embedding"}

# Extract documents and generate embeddings
documents_to_update = collection.find({})  # Fetch all documents

for doc in documents_to_update:
    # Automatically gather text from all fields except ignored ones
    text = " ".join(str(value) for key, value in doc.items() if key not in ignore_fields)

    # Generate embedding and convert to list
    embedding = embedding_model.encode(text).tolist()

    # Update the document with the generated embedding
    collection.update_one(
        {"_id": doc["_id"]},  # Match the document by its _id
        {"$set": {"embeddings": embedding}}  # Add the embedding field to the document
    )

print("Embeddings have been added to all documents (excluding ignored fields).")