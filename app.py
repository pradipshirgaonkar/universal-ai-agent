import streamlit as st
import os
import tempfile
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Page Config
st.set_page_config(page_title="AI Assistant", layout="wide")
st.title("🧠 Universal PDF Assistant")

load_dotenv()
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

# --- Direct Google AI Setup ---
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.error("API Key missing!")

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    
    # Modern stable embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vdb = Chroma.from_documents(docs, embeddings)
    os.remove(tmp_path)
    return vdb

# Sidebar
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    if "vdb" not in st.session_state:
        with st.spinner("Processing PDF..."):
            st.session_state.vdb = process_pdf(uploaded_file)
    
    user_query = st.chat_input("Ask anything about the document...")
    
    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Retrieval
        docs = st.session_state.vdb.similarity_search(user_query, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        # Generation
        full_prompt = f"Context from PDF:\n{context}\n\nQuestion: {user_query}"
        
        with st.chat_message("assistant"):
            try:
                response = model.generate_content(full_prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please upload a PDF to start.")