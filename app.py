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

st.title("🧠 Universal AI Assistant")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # PDF se text nikalne ka sabse stable tarika
    reader = PdfReader(uploaded_file)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()
    
    st.success("PDF analysis complete!")
    
    query = st.text_input("Ask a question about this PDF:")
    
    if query:
        # LLM ko context bhej rahe hain bina kisi extra library ke
        full_prompt = f"PDF Context: {pdf_text[:15000]}\n\nUser Question: {query}"
        
        with st.spinner("Thinking..."):
            try:
                response = model.generate_content(full_prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"API Error: {e}")