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

# NEW STABLE AGENT IMPORTS
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory

# --- UI Setup ---
st.set_page_config(page_title="AI Agent", layout="wide")
st.title("🧠 Universal AI Agent")

load_dotenv()

# --- STEP 1: PDF BRAIN ---
@st.cache_resource
def build_expert_brain():
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        return None, "No PDF Found"
    
    loader = PyPDFLoader(pdf_files[0])
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore, pdf_files[0]

vdb, current_pdf = build_expert_brain()

# --- STEP 2: TOOLS ---
def expert_knowledge(query):
    if vdb:
        results = vdb.similarity_search(query, k=3)
        return "\n".join([r.page_content for r in results])
    return "No PDF uploaded."

tools = [
    Tool(
        name="PDF_Search",
        func=expert_knowledge,
        description="Search information inside the uploaded PDF document."
    )
]

# --- STEP 3: AGENT SETUP ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=api_key,
    transport="rest",  # REST is more stable on Streamlit
    temperature=0.5
)

# Pull the standard ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

# Create Agent and Executor
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=st.session_state.memory, 
    verbose=True,
    handle_parsing_errors=True
)

# --- UI LOGIC ---
if vdb:
    st.success(f"✅ Ready! Document: **{current_pdf}**")
    user_query = st.chat_input("Ask about the document...")
    
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        
        with st.chat_message("assistant"):
            try:
                # Modern .invoke method
                response = agent_executor.invoke({"input": user_query})
                st.write(response["output"])
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.warning("⚠️ Please upload a PDF file to the folder.")