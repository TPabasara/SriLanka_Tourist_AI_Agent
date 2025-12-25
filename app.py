import streamlit as st
import os
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from PIL import Image
import streamlit as st
import base64

img =Image.open('./icons/srilanka.png')
# Set your Google API Key (do this securely in a real app)
load_dotenv()

# Configure LlamaIndex to use Google Gemini
Settings.llm = GoogleGenAI(model_name="models/gemini-2.0-flash")
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")

def add_bg_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def make_header_fixed():
    """Create a fixed header with transparent background."""
    st.markdown(
        """
        <style>
        .fixed-header {
            position: fixed;
            top: 50px;
            left: 0;
            width: 100%;
            height: 18%;
            text-align: center;
            padding: 5px 0;
            backdrop-filter: blur(5px);
            z-index: 999;
        }

        /* Push content below fixed header */
        .main-content {
            margin-top: 130px;
        }

        /* Chat message container */
        [data-testid="stChatMessages"] {
            max-height: 400px;
            overflow-y: auto;
        }

        /* Message spacing */
        [data-testid="stChatMessage"] {
            padding: 6px 12px !important;
            margin-bottom: 6px !important;
        }

        /* Text styling */
        [data-testid="stChatMessageContent"] {
            font-size: 0.95rem !important;
            line-height: 1.25rem !important;
        }

        [data-testid="stChatMessageContent"] {
            font-size: 0.95rem!important;
            line-height: 1.25rem!important;
            font-weight: bold!important;  /* Makes text bold */
            color: #FFFFFF!important;    /* Optional: Forces black text for contrast */
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# --- Page Configuration ---
st.set_page_config(page_title="Sri Lanka Tourist AI", page_icon=img, layout="centered")

make_header_fixed()

st.markdown(
    """
    <div class="fixed-header">
        <h1 style='color: blue;'>Sri Lanka AI Tourist Guide</h1>
        <p style='color: black;'>Ask me anything about locations in Sri Lanka!</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-content">', unsafe_allow_html=True)
add_bg_local("icons/tourisum.png")




# --- Load the Index (RAG) ---
@st.cache_resource
def load_index():
    # Connect to the ChromaDB collection you already created
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_collection("sri_lanka_tourism")
    
    # Create the vector store and index
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index

try:
    index = load_index()
    
    # Create the query engine (your actual agent)
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = index.as_query_engine(streaming=True)

except Exception as e:
    st.error(f"Error loading the knowledge base: {e}")
    st.stop()


# --- Chat Interface Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get new user input
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and stream the AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response_stream = st.session_state.chat_engine.query(prompt)
            
                # Stream the response to the page
                response_placeholder = st.empty()
                full_response = ""
                for token in response_stream.response_gen:
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                # This block runs ONLY if Google sends an error (like the 429 quota error)
                error_message = "⚠️ I am currently overwhelmed with requests. As a free service, I have limitations. Please wait about a minute and try asking again!"
                st.error(error_message)
                # Optional: Print the actual error to your console logs for debugging
                print(f"DEBUG: Gemini API Error encountered: {e}")