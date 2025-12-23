import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
# MODERN IMPORT: Deprecation error fix 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
import tempfile

# API Key
load_dotenv()
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Stable Model Initialization (Version v1 is key here)
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=api_key,
        temperature=0.3,
        version="v1" # v1beta   bypass 
    )
except Exception as e:
    st.error(f"LLM Setup Error: {e}")

# PDF Processing function
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    
    # Modern Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vdb = Chroma.from_documents(docs, embeddings)
    os.remove(tmp_path)
    return vdb

# UI Sidebar & Chat
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    if "vdb" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Analyzing PDF..."):
            st.session_state.vdb = process_pdf(uploaded_file)
            st.session_state.file_name = uploaded_file.name
            st.session_state.chat_history = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
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
                prompt = f"Context: {context}\n\nQuestion: {user_query}"
                
                response = llm.invoke(prompt)
                st.write(response.content)
                st.session_state.chat_history.append(AIMessage(content=response.content))
            except Exception as e:
                st.error(f"Query Error: {e}")
else:
    st.info("Please upload a PDF to start.")