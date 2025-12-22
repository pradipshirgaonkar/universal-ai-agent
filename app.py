import streamlit as st
import os
import glob 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import hub
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# --- UI Setup ---
st.set_page_config(page_title="Universal AI Agent", layout="wide")
st.title("🧠 Universal Subject-Matter Expert Agent")
st.sidebar.info("Just convert the PDF file into a folder, and ask me whatever you want to know about that topic.!")

load_dotenv()

# --- STEP 1: DYNAMIC PDF DISCOVERY ---
@st.cache_resource
def build_expert_brain():
    # Folder mein jo bhi pehli PDF milegi, use utha lega
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        return None, "File not found!"
    
    active_pdf = pdf_files[0] # Pehli PDF ko subject banao
    loader = PyPDFLoader(active_pdf)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore, active_pdf

vdb, current_subject = build_expert_brain()

# --- STEP 2: BRAIN & MEMORY ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    system_instruction="You are a helpful incurrent subject assistant. Always respond in the same language the user uses to ask the question (Hindi, English, or Hinglish)."
)

# --- STEP 3: THE EXPERT TOOL ---
def expert_knowledge(query):
    results = vdb.similarity_search(query, k=3)
    return "\n".join([r.page_content for r in results])

tools = [
    Tool(
        name="Expert_Knowledge_Base",
        func=expert_knowledge,
        description=f"Use this tool to answer questions about {current_subject}."
    )
]

# --- STEP 4: AGENT ENGINE ---
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=st.session_state.memory
)

# --- UI LOGIC ---
if vdb:
    st.success(f"✅ Current Expertise: **{current_subject}**")
    user_query = st.chat_input("Ask me anything about this subject...")
    
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        with st.chat_message("assistant"):
            response = agent.run(input=user_query)
            st.write(response)
else:
    st.error("⚠️ Please upload or add a PDF file in the project folder!")