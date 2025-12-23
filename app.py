import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import os
from dotenv import load_dotenv

load_dotenv()
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

st.title("🧠 PDF Quick Chat")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # 1. Direct PDF Text Extraction (No heavy libraries)
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    st.success("PDF Loaded!")
    
    user_query = st.chat_input("Ask about the PDF...")
    
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        
        prompt = f"Context: {text[:10000]}\n\nQuestion: {user_query}"
        
        with st.chat_message("assistant"):
            try:
                response = model.generate_content(prompt)
                st.write(response.text)
            except Exception as e:
                st.error(f"Error: {e}")