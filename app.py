__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import glob 
from dotenv import load_dotenv

# Essential AI Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# MODERN AGENT IMPORTS (No more initialize_agent error)
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory

# --- UI Setup ---
st.set_page_config(page_title="Universal AI Agent", layout="wide")
st.title("🧠 Universal AI Agent (PDF Knowledge Base)")

load_dotenv()

# --- STEP 1: DYNAMIC PDF DISCOVERY ---
@st.cache_resource
def build_expert_brain():
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        return None, "No PDF"
    
    active_pdf = pdf_files[0] 
    loader = PyPDFLoader(active_pdf)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore, active_pdf

vdb, current_subject = build_expert_brain()

# --- STEP 2: TOOLS SETUP ---
def expert_knowledge(query):
    if vdb:
        results = vdb.similarity_search(query, k=3)
        return "\n".join([r.page_content for r in results])
    return "No PDF uploaded."

tools = [
    Tool(
        name="PDF_Knowledge_Base",
        func=expert_knowledge,
        description=f"Use this to answer questions about the document: {current_subject}."
    )
]

# --- STEP 3: AGENT & MEMORY ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=api_key,
    transport="rest", # gRPC error fix
    temperature=0.5
)

# Pull modern ReAct prompt
prompt = hub.pull("hwchase17/react")

# Create the Agent
agent = create_react_agent(llm, tools, prompt)

# Create the Executor
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=st.session_state.memory, 
    verbose=True,
    handle_parsing_errors=True
)

# --- UI LOGIC ---
if vdb:
    st.success(f"✅ Loaded: **{current_subject}**")
    
    user_query = st.chat_input("Ask me anything about the PDF...")
    
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        
        with st.chat_message("assistant"):
            # Use .invoke instead of .run
            result = agent_executor.invoke({"input": user_query})
            st.write(result["output"])
else:
    st.error("⚠️ Please add a PDF file in the folder!")