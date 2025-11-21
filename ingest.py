import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# Set your Google API Key
load_dotenv()

# Configure LlamaIndex to use Google Gemini
Settings.llm = GoogleGenAI(model_name="models/gemini-1.5-flash")
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")

# 1. Load Documents
# This reads all files from your ./data folder
print("Loading documents...")
documents = SimpleDirectoryReader("./data").load_data()

# 2. Create a ChromaDB client and collection
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("sri_lanka_tourism")

# 3. Create a Vector Store and Storage Context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 4. Create the Index
# This does all the work:
# - Splits documents into chunks
# - Creates "embeddings" (vector numbers) for each chunk
# - Stores them in your ChromaDB
print("Creating index and storing embeddings...")
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
print("Done. Your knowledge base is ready.")