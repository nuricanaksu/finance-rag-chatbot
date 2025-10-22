# 💠 FinSage AI — Financial Insight Assistant  
*A Retrieval-Augmented Educational Finance Chatbot powered by Gemini 2.0 Flash*

[![Streamlit App](https://img.shields.io/badge/🚀_Live_App-Streamlit-brightgreen?logo=streamlit)](https://finsageai.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Gemini_2.0_Flash-ff69b4?logo=google)](https://ai.google.dev/gemini-api)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

**FinSage AI** is a **Retrieval-Augmented Generation (RAG)**-based educational chatbot  
designed to provide conceptual explanations for finance and economics.  

It combines **vector search**, **semantic retrieval**, and **Google Gemini 2.0 Flash**  
to deliver accurate, context-aware answers from the *Finance-Alpaca* dataset.

---

## 🧠 Tech Stack

| Component | Description |
|------------|-------------|
| **Frontend** | Streamlit UI with custom dark theme and chat layout |
| **Backend** | Python (RAG pipeline + FAISS + Gemini 2.0 Flash) |
| **LLM** | Google Gemini 2.0 Flash |
| **Embeddings** | BAAI/bge-small-en-v1.5 |
| **Vector Store** | FAISS (Flat Inner Product index) |
| **Dataset** | `gbharti/finance-alpaca` (via Hugging Face) |
| **Deployment** | Streamlit Cloud |

---

## 🧩 Key Features

✅ Context-aware financial question answering  
✅ Uses RAG (Retrieval-Augmented Generation) pipeline  
✅ Gemini 2.0 Flash for reasoning & summarization  
✅ Modern chat interface with typing animation  
✅ Info modal explaining dataset & limitations  
✅ Ready-to-deploy structure (secure API handling)

---

## 🧱 How It Works — RAG Pipeline Explained

FinSage AI follows the **Retrieval-Augmented Generation (RAG)** architecture:

1. **Query Input (User)**  
   The user asks a financial question (e.g., *“What is compound interest?”*).

2. **Retrieval Step (FAISS + Embeddings)**  
   The system searches a **vector database** (FAISS) containing sentence embeddings  
   from the **Finance-Alpaca** dataset using **BAAI/bge-small-en-v1.5** model.  
   → Top-5 semantically similar text chunks are retrieved.

3. **Prompt Construction**  
   These retrieved passages are appended as *context* to the user’s query.

4. **Generation (Gemini 2.0 Flash)**  
   The contextual prompt is sent to **Gemini 2.0 Flash**, which generates  
   a focused, educational, and context-grounded answer.

5. **UI Rendering (Streamlit)**  
   The result is displayed dynamically inside a styled chat interface  
   with real-time typing simulation for natural interaction.

---

## 🧬 Architecture Diagram

<p align="center">
  <img src="assets/finsage_rag_diagram.png" alt="RAG Architecture" width="720">
</p>

> **Diagram Explanation:**  
> - **Embeddings:** Precomputed vectors stored in FAISS index.  
> - **Retriever:** Finds the most semantically relevant contexts.  
> - **Generator (Gemini):** Synthesizes human-like, contextual answers.  
> - **Streamlit UI:** Handles chat flow and response visualization.

---

## ⚙️ Installation & Local Run

```bash
# 1️⃣ Clone the repository
git clone https://github.com/nuricanaksu/finance-rag-chatbot.git
cd finance-rag-chatbot

# 2️⃣ Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Add your Google API Key (securely)
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# 5️⃣ Run Streamlit app
streamlit run app.py
💡 The app will open at http://localhost:8501

🔐 API Key Security
To keep your API key secure:

Store it in .env or Streamlit Cloud Secrets Manager (.streamlit/secrets.toml)

Never push .env to GitHub (add it to .gitignore)

The app reads the key dynamically using dotenv or st.secrets["GOOGLE_API_KEY"]

🧾 Notebook Integration (for grading & documentation)
The project also includes .ipynb notebooks (GenAI_Section_1.ipynb, GenAI_Section_2.ipynb)
that document the entire workflow step-by-step in Markdown cells:

Section	Content
Section 1	RAG theory, Gemini setup, data preprocessing
Section 2	Embeddings, FAISS indexing, chatbot integration

These notebooks ensure compliance with:

“All technical explanations must be included within Markdown cells or comments.”

🎥 Demo Video
🎬 Watch FinSage AI in Action
(Explains dataset loading, FAISS retrieval, and Gemini answer generation)

<p align="center"> <img src="assets/demo.gif" alt="FinSage Demo" width="700"> </p>
⚠️ Limitations
Educational purpose only (no real-time or personalized advice)

Finance-Alpaca dataset = static (pre-2023 data)

Model cannot access internet or live market data

📜 License
This project is licensed under the MIT License.
You are free to fork, modify, and deploy with attribution.

<p align="center"> <sub>💡 Built with passion by <a href="https://github.com/nuricanaksu">Nuri Aksu</a> — Powered by Gemini 2.0 Flash</sub> </p> ```