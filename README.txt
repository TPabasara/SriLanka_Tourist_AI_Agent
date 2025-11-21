# Sri Lanka Tourist AI Guide (RAG Agent)
A specialized AI agent designed to assist tourists visiting Sri Lanka with verified, hallucination-free travel information.

## 📖 Project Overview
Unlike generic LLMs that often "hallucinate" incorrect facts about ticket prices or opening hours, this application uses Retrieval-Augmented Generation (RAG). It provides accurate answers regarding historical significance, logistics, and travel tips for major Sri Lankan destinations (e.g., Sigiriya, Yala National Park, Ella) by referencing a curated knowledge base.

## ✨ Key Features
🧠 RAG Architecture: Connects a Large Language Model to a custom, curated dataset of text documents for high factual accuracy.

⚡ Real-Time Streaming: Implements token-streaming to provide a fast, "ChatGPT-like" conversational experience.

💬 Context Awareness: Maintains chat history using session state, allowing users to ask follow-up questions naturally.

📚 Source Attribution: Answers are derived directly from indexed travel research data, ensuring reliability.

## 🛠️ Technical Stack
Language: Python 3.10+

Framework: LlamaIndex (v0.10+) for RAG orchestration and data ingestion.

LLM (The Brain): Google Gemini Pro (gemini-1.5-flash) via Google GenAI API.

Embeddings: Google GenAI Embeddings (models/text-embedding-004) for vectorizing text.

Vector Database: chroma_db for storing and retrieving high-dimensional vector data.

Frontend: streamlit for the web interface, state management, and cloud deployment.

Environment: Virtual environments (venv) and python-dotenv for secure API key management.

## ⚙️ How It Works
Ingestion Pipeline (ingest.py)

A script reads raw text files containing curated travel data.

It chunks the text and converts it into vector embeddings using Google's embedding model.

These vectors are stored locally in a persistent ChromaDB vector store.

Inference Engine (app.py)

When a user asks a question, the app queries ChromaDB for the most relevant context chunks.

It injects that specific context into a system prompt.

It sends the augmented prompt to the Gemini LLM to generate a factual, natural language response.

## 🚀 Getting Started
**Clone the repository:**bash git clone https://github.com/yourusername/sri-lanka-tourist-ai.git

Install dependencies: pip:bash pip install -r requirements.txt

Set up API keys: Create a .env file and add your Google API key: GOOGLE_API_KEY="your_key_here"

Build the database: python ingest.py

Run the app: streamlit run app.py