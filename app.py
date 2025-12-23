import streamlit as st
import os
import glob 
from dotenv import load_dotenv

import langchain
# Naye LangChain versions ke liye ye zaroori hai
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain import hub

# Baki community imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun # Iska path double check karein
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- UI Setup ---
st.set_page_config(page_title="Universal AI Agent", layout="wide")
st.title("🧠 Universal AI Agent (PDF + Web)")

load_dotenv()

@st.cache_resource
def build_expert_brain():
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        return None, "No PDF"
    active_pdf = pdf_files[0] 
    loader = PyPDFLoader(active_pdf)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore, active_pdf

vdb, current_subject = build_expert_brain()

# --- Memory Setup ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Agent Engine ---
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

def pdf_qa(query):
    if vdb:
        return "\n".join([r.page_content for r in vdb.similarity_search(query, k=3)])
    return "No PDF data."

search = DuckDuckGoSearchRun()
tools = [
    Tool(name="PDF_Search", func=pdf_qa, description="Search info in the PDF."),
    Tool(name="Web_Search", func=search.run, description="Search info on the internet.")
]

# React Prompt pull karna (Modern Way)
prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=True, handle_parsing_errors=True)

# --- UI ---
if vdb:
    st.success(f"Expertise: {current_subject}")
    user_query = st.chat_input("Ask me anything...")
    if user_query:
        with st.chat_message("user"): st.write(user_query)
        with st.chat_message("assistant"):
            # 'input' key use karein naye executor ke liye
            response = agent_executor.invoke({"input": user_query})
            st.write(response["output"])
else:
    st.warning("Please upload a PDF to start.")