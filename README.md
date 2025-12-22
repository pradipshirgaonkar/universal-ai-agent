# 🧠 Universal Subject-Matter Expert AI Agent
### *Autonomous Agentic RAG Pipeline with Conversational Memory*

An advanced **Agentic RAG (Retrieval-Augmented Generation)** system built to eliminate AI hallucinations and provide grounded, data-driven answers from private or latest PDF documents. 



## 🚀 Overview
Standard LLMs like ChatGPT have a knowledge cutoff. This project bridges that gap by allowing users to "plug in" any PDF (e.g., Budget 2025, Medical Reports, or Technical Manuals) and instantly turn the AI into a domain expert on that subject.

## ✨ Key Features
- **Dynamic Knowledge Injection:** The system automatically builds a knowledge base from any PDF placed in the directory.
- **Agentic Reasoning (ReAct):** Uses LangChain's reasoning logic to autonomously decide between searching the internal document or the live web.
- **Conversational Memory:** Remembers past interactions within the session for seamless follow-up questions.
- **Multilingual Support:** Fully capable of understanding and responding in English, Hindi, and Hinglish.

## 🛠️ Technical Tech Stack
- **Orchestration:** LangChain
- **LLM:** Google Gemini 1.5 Pro
- **Vector Database:** ChromaDB (Persistent Storage)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local & Fast)
- **UI Framework:** Streamlit
- **Environment Management:** Python Dotenv (Security)

## 📁 Project Structure
```text
paiAGENT/
├── .env                # Private API Keys (Hidden via .gitignore)
├── .gitignore          # Prevents sensitive data from being pushed to GitHub
├── app.py              # Main Application Logic & UI
├── requirements.txt    # Project Dependencies
├── README.md           # Professional Documentation
└── [Subject].pdf       # Any PDF file (e.g., Budget_2025.pdf)
⚙️ How to Setup & Run
Clone the repository:

Bash

git clone [https://github.com/pradipshirgaonkar/universal-ai-agent.git](https://github.com/pradipshirgaonkar/universal-ai-agent.git)
cd universal-ai-agent
Install necessary libraries:

Bash

pip install -r requirements.txt
Configure API Key: Create a .env file and add your Gemini API Key:

Plaintext

GOOGLE_API_KEY=your_gemini_api_key_here
Launch the Agent:

Bash

streamlit run app.py
🧠 System Architecture
Document Processing: PDF is split into overlapping chunks for better context retention.

Vectorization: HuggingFace transforms text into mathematical vectors stored in ChromaDB.

Reasoning: Upon receiving a query, the Agent analyzes the intent. If the answer is in the PDF, it uses Budget_Search. If not, it triggers Web_Search.

💼 Business Use Cases
Enterprise HR/Policy Support: Instantly query complex company policy documents.

Financial Analysis: Analyze latest budget documents or quarterly reports.

Legal/Tech Support: Rapidly extract info from long contracts or technical manuals.

Developed by: Pradip Kamble

Role: AI Project Manager & Developer