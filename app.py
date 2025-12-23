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

# API Key check
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# --- PDF Processing Function ---
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vdb = Chroma.from_documents(docs, embeddings)
    os.remove(tmp_path) # Clean up temporary file
    return vdb

# --- Sidebar: File Upload ---
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    if "vdb" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Analyzing PDF..."):
            st.session_state.vdb = process_pdf(uploaded_file)
            st.session_state.file_name = uploaded_file.name
        st.sidebar.success("File Processed!")

    # --- Chat Interface ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

    user_query = st.chat_input("Ask about the PDF...")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.write(user_query)
        
        with st.chat_message("assistant"):
            # Retrieval logic
            docs = st.session_state.vdb.similarity_search(user_query, k=3)
            context = "\n".join([d.page_content for d in docs])
            prompt = f"Context: {context}\n\nQuestion: {user_query}"
            
            response = llm.invoke(prompt).content
            st.write(response)
            st.session_state.chat_history.append(AIMessage(content=response))
else:
    st.info("Please upload a PDF file from the sidebar to start.")