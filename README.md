# 💠 FinSage AI — Financial Insight Assistant  
*A Retrieval-Augmented Educational Finance Chatbot*

[![Streamlit App](https://img.shields.io/badge/🚀_Live_App-Streamlit-brightgreen?logo=streamlit)](https://finsageai.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Gemini_2.0_Flash-ff69b4?logo=google)](https://ai.google.dev/gemini-api)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

**FinSage AI** is a **Retrieval-Augmented Generation (RAG)** based financial assistant.  
It is designed to provide **conceptual financial insights** trained on the *Finance-Alpaca* dataset —  
a specialized version of the Stanford Alpaca dataset adapted for finance and economics.

This project demonstrates how **vector search, semantic retrieval, and generative AI**  
can be combined to build a domain-focused, educational chatbot.

---

## 🧠 Tech Stack

| Component | Description |
|------------|-------------|
| **Frontend** | Streamlit UI with modern dark theme & dynamic chat bubbles |
| **Backend** | Python (Fast Retrieval + Gemini API) |
| **Model** | Google Gemini 2.0 Flash |
| **Embeddings** | BAAI/bge-small-en-v1.5 |
| **Vector Store** | FAISS |
| **Dataset** | `gbharti/finance-alpaca` (HuggingFace) |

---

## 🎥 Demo Video

👉 [Watch Full Demo on YouTube](https://your-video-link.com)  
*(Show the chatbot UI, example questions, and dataset loading flow)*  

<p align="center">
  <img src="assets/demo.gif" alt="FinSage Demo" width="700">
</p>

---

## 🧩 Key Features

✅ Retrieval-Augmented response generation  
✅ Context-aware finance Q&A  
✅ Modern chat UI with typing animation  
✅ Info modal (limitations, dataset scope)  
✅ Example questions for guided testing  
✅ Clean deployment with Streamlit Cloud  

---

## ⚙️ Installation & Run Locally

```bash
# 1️⃣ Clone the repository
git clone https://github.com/nuricanaksu/finance-rag-chatbot.git
cd finance-rag-chatbot

# 2️⃣ Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or on Windows: .venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Add your Google API Key
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# 5️⃣ Run Streamlit
streamlit run app.py

Then open http://localhost:8501
 to view your app.

⚠️ Limitations

FinSage AI is purely educational.
It does not provide real-time or personalized financial advice.
Data is static and limited to the Finance-Alpaca dataset (pre-2023).

📜 License

This project is licensed under the MIT License
.
Feel free to fork, learn, and adapt — just keep it open source ❤️

<p align="center"> <sub>Created with 💡 by <a href="https://github.com/nuricanaksu">Nuri Aksu</a></sub> </p> ```