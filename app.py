import streamlit as st
import os
import glob 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun # Naya Tool

# --- UI Setup ---
st.set_page_config(page_title="Universal AI Agent", layout="wide")
st.title("🧠 Universal AI Agent (PDF + Web Search)")

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
# 1. PDF Tool
def expert_knowledge(query):
    if vdb:
        results = vdb.similarity_search(query, k=3)
        return "\n".join([r.page_content for r in results])
    return "No PDF uploaded."

# 2. Web Search Tool
search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="PDF_Knowledge_Base",
        func=expert_knowledge,
        description=f"Use this FIRST to answer questions about {current_subject}."
    ),
    Tool(
        name="Web_Search",
        func=search.run,
        description="Use this ONLY if you cannot find the answer in the PDF or for current events/news."
    )
]

# --- STEP 3: AGENT & MEMORY ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=api_key,
    temperature=0.5,
    system_instruction="You are a smart assistant. First check the PDF tool. If the info is missing, use Web Search."
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=st.session_state.memory,
    handle_parsing_errors=True 
)

# --- UI LOGIC ---
if vdb:
    st.success(f"✅ Loaded: **{current_subject}** | 🌐 Web Search: **Active**")
    
    user_query = st.chat_input("Ask about the PDF or anything else...")
    
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        
        with st.chat_message("assistant"):
            response = agent.run(input=user_query)
            st.write(response)
else:
    st.error("⚠️ Please add a PDF file in the folder!")