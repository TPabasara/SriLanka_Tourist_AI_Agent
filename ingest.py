import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding


load_dotenv()

Settings.embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-2")
Settings.llm = GoogleGenAI(model="models/gemini-3.5-flash", temperature=0.2)


print("Loading documents...")
documents = SimpleDirectoryReader("./data").load_data()


db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("sri_lanka_tourism")


vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


print("Creating index and storing embeddings...")
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
print("Done. Your knowledge base is ready.")