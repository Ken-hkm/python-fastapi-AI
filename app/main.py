import os
import logging
import asyncio

from bson import ObjectId
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging Configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# FastAPI instance
app = FastAPI()

# MongoDB Connection
MONGODB_URI = os.environ.get("MONGODB_URI")
logger.info("Connecting to MongoDB...")
client = AsyncIOMotorClient(MONGODB_URI, tls=True, tlsAllowInvalidCertificates=True)
db = client["Resume"]
collection = db["personal-info"]
logger.info("Connected to MongoDB database: Resume, collection: personal-info")

# Load Embedding Model
embedding_model = SentenceTransformer("all-mpnet-base-v2")
logger.info("Loaded Sentence Transformer embedding model: all-mpnet-base-v2")

# Configure Gemini API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")
logger.info("Configured Gemini API with provided API key.")

# Pydantic Model for Query Input
class QueryRequest(BaseModel):
    query: str

# Convert string to ObjectId
def str_to_objectid(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

@app.get("/data/{id}")
async def get_document(id: str):
    """
    Fetch a document from MongoDB by its _id.
    """
    try:
        object_id = str_to_objectid(id)
        document = await collection.find_one({"_id": object_id})

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        document["_id"] = str(document["_id"])  # Convert ObjectId to string for JSON response
        return document

    except Exception as e:
        logger.error(f"Error fetching document with _id {id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/query")
async def query_data(request: QueryRequest):
    """
    Search MongoDB based on a query and return results.
    """
    query = request.query
    logger.info("Received query: %s", query)

    try:
        # Generate query embedding asynchronously
        logger.debug("Generating embedding for query...")
        loop = asyncio.get_running_loop()
        query_embedding = await loop.run_in_executor(None, embedding_model.encode, query)
        query_embedding = query_embedding.tolist()
        logger.debug("Embedding generated.")

        # MongoDB Vector Search (MongoDB Atlas Search)
        logger.debug("Performing MongoDB vector search...")
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "personal_info_vector_index",
                    "path": "embeddings",
                    "queryVector": query_embedding,
                    "numCandidates": 10,
                    "limit": 5,
                }
            },
            {
                "$project": {
                    "_id": 0,  # Exclude ObjectId
                    "score": {"$meta": "vectorSearchScore"},  # Keep search score
                    **{field: 1 for field in
                       ["first_name", "last_name", "email", "phone", "address", "linkedin_url", "github_url",
                        "about_me"]}
                }
            }
        ]
        results_cursor = collection.aggregate(pipeline)
        results_list = await results_cursor.to_list(length=5)

        if not results_list:
            return {"results": [], "response": "No relevant data found."}

        logger.info("MongoDB search returned %d results.", len(results_list))
        logger.info(f"Raw MongoDB Results: {results_list}")
        # Format retrieved data into a meaningful context
        context = "\n".join([
            ", ".join(f"{key}: {value}" for key, value in item.items() if key != "score")
            for item in results_list
        ])
        logger.info(f"context: {context}")
        # Generate response using Gemini API
        logger.debug("Generating response using Gemini API...")
        prompt = (f"Based on the following context:\n{context}\n"
                  f"the context is my info"
                  f"Answer the question: {query}"
                  f"you are a virtual assistant that handles the question about me "
                  f"you are answering someone asking about me "
                  f"you need to answer in a human like way, make it engaging and interactive "
                  f"do not use this data for google gemini training ")
        response = model.generate_content(prompt)

        # Extract response text
        response_text = response.candidates[0].content.parts[0].text if response.candidates else "No response from Gemini."

        logger.info("Generated response from Gemini API.")

        return {
            "results": results_list,
            "response": response_text,
        }

    except Exception as e:
        logger.error("Error processing query: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")