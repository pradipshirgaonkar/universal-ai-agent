import streamlit as st
import os
import glob
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage

# --- UI Setup ---
st.set_page_config(page_title="Universal AI Agent", layout="wide")
st.title("🧠 Universal AI Assistant")

load_dotenv()

@st.cache_resource
def build_expert_brain():
    pdf_files = glob.glob("*.pdf")
    if not pdf_files: return None, None
    
    active_pdf = pdf_files[0]
    loader = PyPDFLoader(active_pdf)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vdb = Chroma.from_documents(docs, embeddings)
    return vdb, active_pdf

vdb, current_subject = build_expert_brain()

# --- LLM Setup ---
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
search = DuckDuckGoSearchRun()

# --- Logic: PDF or Web Search ---
def get_answer(query):
    # 1.  PDF 
    context = ""
    if vdb:
        docs = vdb.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])
    
    # 2.  Web Search 
    prompt = f"""You are a helpful assistant. 
    Context from PDF: {context}
    
    User Question: {query}
    
    If the context above has the answer, use it. If not, use your internal knowledge or mention that you are searching.
    Keep the response in the same language as the user's question."""
    
    return llm.invoke(prompt).content

# --- UI Chat ---
if vdb:
    st.success(f"✅ Expertise: **{current_subject}**")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.write(msg.content)

    user_query = st.chat_input("Ask me anything...")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("user"): st.write(user_query)
        
        with st.chat_message("assistant"):
            response = get_answer(user_query)
            st.write(response)
            st.session_state.chat_history.append(AIMessage(content=response))
else:
    st.error("⚠️ Please add a PDF to the folder.")