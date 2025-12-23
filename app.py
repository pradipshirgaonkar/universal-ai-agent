import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os, tempfile
from dotenv import load_dotenv

st.set_page_config(page_title="Universal AI Assistant", layout="wide")
st.title("🧠 Personal AI Assistant")

load_dotenv()
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

# --- Direct Google SDK Setup (Stable) ---
if api_key:
    genai.configure(api_key=api_key)
    # Yeh hamesha stable v1 endpoint use karta hai
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.error("API Key not found in Secrets!")

def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        path = tmp.name
    
    loader = PyPDFLoader(path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    
    # Stable Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vdb = Chroma.from_documents(docs, embeddings)
    os.remove(path)
    return vdb

# UI
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    if "vdb" not in st.session_state:
        with st.spinner("Analyzing PDF..."):
            st.session_state.vdb = process_pdf(uploaded_file)
    
    user_query = st.chat_input("Ask about the PDF...")
    if user_query:
        with st.chat_message("user"): st.write(user_query)
        
        # Retrieval
        docs = st.session_state.vdb.similarity_search(user_query, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        # Generation via Official Google SDK (Bypassing LangChain error)
        prompt = f"Use this context to answer: {context}\n\nQuestion: {user_query}"
        
        with st.chat_message("assistant"):
            try:
                response = model.generate_content(prompt)
                st.write(response.text)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Upload a PDF to start.")