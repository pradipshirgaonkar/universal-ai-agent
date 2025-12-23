import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
import tempfile

# --- UI Setup ---
st.set_page_config(page_title="AI PDF Assistant", layout="wide")
st.title("🧠 Personal AI PDF Assistant")

load_dotenv()

# API Key check from Streamlit Secrets or .env
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ Google API Key not found! Please add it to Streamlit Secrets.")
    st.stop()

# LLM Setup with safety settings
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # Agar ye fail ho toh "models/gemini-1.5-flash" try karein
        google_api_key=api_key,
        temperature=0.3,
    )
except Exception as e:
    st.error(f"Failed to initialize Gemini: {e}")
    st.stop()

# --- PDF Processing ---
def process_pdf(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(data)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vdb = Chroma.from_documents(docs, embeddings)
        os.remove(tmp_path)
        return vdb
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

# --- UI Sidebar ---
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    if "vdb" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Analyzing PDF..."):
            st.session_state.vdb = process_pdf(uploaded_file)
            st.session_state.file_name = uploaded_file.name
            st.session_state.chat_history = [] # Reset chat for new file

    # Chat UI
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.write(msg.content)

    user_query = st.chat_input("Ask about the PDF...")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("user"): st.write(user_query)
        
        with st.chat_message("assistant"):
            try:
                # Context Retrieval
                docs = st.session_state.vdb.similarity_search(user_query, k=3)
                context = "\n".join([d.page_content for d in docs])
                
                # Call Gemini
                response = llm.invoke(f"Context: {context}\n\nQuestion: {user_query}")
                st.write(response.content)
                st.session_state.chat_history.append(AIMessage(content=response.content))
            except Exception as e:
                st.error(f"Gemini API Error: {e}")
                st.info("Tip: Check if your API Key is valid and you have quota left.")
else:
    st.info("👋 Hi! Please upload a PDF file from the sidebar to start chatting.")