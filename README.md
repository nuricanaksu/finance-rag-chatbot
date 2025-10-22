# 💠 FinSage AI — Financial Insight Assistant  
*A Retrieval-Augmented Educational Finance Chatbot powered by Gemini 2.0 Flash*

[![Streamlit App](https://img.shields.io/badge/🚀_Live_App-Streamlit-brightgreen?logo=streamlit)](https://finsageai.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Gemini_2.0_Flash-ff69b4?logo=google)](https://ai.google.dev/gemini-api)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

**FinSage AI** is a **Retrieval-Augmented Generation (RAG)**-based educational chatbot  
designed to explain fundamental concepts in **finance and economics**.  
It combines **vector search**, **semantic retrieval**, and **Google Gemini 2.0 Flash**  
to generate accurate, context-grounded insights from the *Finance-Alpaca* dataset.

---

## 🧠 Tech Stack

| Component | Description |
|------------|-------------|
| **Frontend** | Streamlit UI with custom dark theme & chat interface |
| **Backend** | Python (RAG pipeline + FAISS + Gemini 2.0 Flash) |
| **LLM** | Google Gemini 2.0 Flash |
| **Embeddings** | BAAI/bge-small-en-v1.5 |
| **Vector Store** | FAISS (Flat Inner Product Index) |
| **Dataset** | `gbharti/finance-alpaca` from Hugging Face |
| **Deployment** | Streamlit Cloud |

---

## 🧩 Key Features

✅ Retrieval-Augmented Generation (RAG) workflow  
✅ Context-aware financial question answering  
✅ Typing animation for natural chat effect  
✅ Info modal explaining dataset and limitations  
✅ Example question suggestions  
✅ Secure API key handling for deployment  

---

## ⚙️ Installation & Local Run

### 🧱 1. Clone the repository
```bash
git clone https://github.com/nuricanaksu/finance-rag-chatbot.git
cd finance-rag-chatbot
🧰 2. Create a virtual environment
bash
Kodu kopyala
python -m venv .venv
source .venv/bin/activate     # or on Windows: .venv\Scripts\activate
📦 3. Install dependencies
bash
Kodu kopyala
pip install -r requirements.txt
🔑 4. Add your Google API Key (securely)
bash
Kodu kopyala
echo "GOOGLE_API_KEY=your_api_key_here" > .env
▶️ 5. Run the Streamlit app
bash
Kodu kopyala
streamlit run app.py
After launch, open http://localhost:8501 in your browser.

🔐 API Key Security
Your API key should never be pushed to GitHub.

To secure it:

Store locally in .env

Or use Streamlit Cloud Secrets Manager (.streamlit/secrets.toml)

Example for Streamlit:

toml
Kodu kopyala
GOOGLE_API_KEY = "your_api_key_here"
🧾 Notebook Integration (for grading & documentation)
The project includes .ipynb notebooks explaining the workflow step-by-step.

Section	Description
Section 1	RAG theory, Gemini setup, data preprocessing
Section 2	Embeddings, FAISS indexing, chatbot integration

These notebooks ensure compliance with the guideline:

“All technical explanations must be included within Markdown cells or comments.”

🎥 Demo Video
🎬 Watch FinSage AI in Action
Explains dataset loading, FAISS retrieval, and Gemini-based generation.

(Replace the link above once you upload your demo video)

⚠️ Limitations
Educational purpose only (not real-time or personalized financial advice)

Dataset is static and pre-2023

No live market or cryptocurrency data access

📜 License
This project is licensed under the MIT License.
You are free to fork, modify, and deploy — with attribution.

<p align="center"> <sub>💡 Built with passion by <a href="https://github.com/nuricanaksu">Nuri Aksu</a> — Powered by Gemini 2.0 Flash</sub> </p> ```