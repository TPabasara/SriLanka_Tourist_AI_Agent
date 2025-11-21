Sri Lanka Tourist AI Guide (RAG Agent)
Project Overview A specialized AI agent designed to assist tourists visiting Sri Lanka. Unlike generic LLMs that often "hallucinate" incorrect facts, this application uses Retrieval-Augmented Generation (RAG) to provide accurate, verified answers regarding ticket prices, opening hours, historical significance, and travel tips for major Sri Lankan destinations (e.g., Sigiriya, Yala National Park, Ella).

Key Features

RAG Architecture: Connects a Large Language Model to a custom, curated dataset of PDF/Text documents for high accuracy.

Real-Time Streaming: Implements token-streaming to provide a fast, "ChatGPT-like" conversational experience.

Context Awareness: Maintains chat history using session state, allowing follow-up questions.

Source Attribution: Answers are derived directly from indexed travel research data.

Technical Stack

Language: Python 3.10+

Framework: LlamaIndex (v0.10+) for RAG orchestration and data ingestion.

LLM (The Brain): Google Gemini Pro (gemini-1.5-flash) via Google GenAI API.

Embeddings: Google GenAI Embeddings (models/text-embedding-004) for vectorizing text.

Vector Database: ChromaDB (Persistent Client) for storing and retrieving high-dimensional vector data.

Frontend: Streamlit for the web interface, state management, and cloud deployment.

Environment: Virtual environments (venv) and python-dotenv for secure API key management.

How It Works

Ingestion Pipeline (ingest.py): A script reads raw text files containing travel data, chunks them, converts them into vector embeddings using Google's model, and stores them locally in ChromaDB.

Inference Engine (app.py): When a user asks a question, the app queries ChromaDB for the most relevant context, injects that context into a prompt, and sends it to the Gemini LLM to generate a factual response.