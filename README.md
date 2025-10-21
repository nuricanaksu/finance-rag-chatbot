# **FinSage AI — Financial Insight Assistant**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen)
![Model](https://img.shields.io/badge/Model-Gemini%202.0%20Flash-ff69b4)
![License](https://img.shields.io/badge/License-MIT-yellow)

**FinSage AI** is an intelligent financial insight assistant built with a Retrieval-Augmented Generation (RAG) architecture.  
It combines semantic retrieval and generative reasoning to deliver educational explanations around markets, investments, and economic concepts.

🔗 **Live Application:** [https://finsageai.streamlit.app/](https://finsageai.streamlit.app/)

---

## **Overview**

FinSage AI integrates semantic retrieval and Google’s Gemini 2.0 Flash model to provide concise, context-aware responses to finance-related questions.  
The system retrieves relevant knowledge from the *Finance-Alpaca* dataset and generates structured, educational responses designed for learning — not financial advice.

Use cases include:
- Understanding financial terminology  
- Learning about investments and markets  
- Exploring economic indicators  
- Comparing financial instruments  

---

## **System Design**

### 🧱 Architecture
1. **Embedding Generation:** Encodes financial text using the `BAAI/bge-small-en-v1.5` model.  
2. **Vector Search:** Uses **FAISS** for efficient similarity search over embedded data.  
3. **Contextual Retrieval:** Selects the top-k most relevant contexts for each query.  
4. **Response Generation:** Combines retrieved context with **Gemini 2.0 Flash** for structured, human-like answers.  
5. **Interface:** Fully implemented with **Streamlit**, styled for clarity and modern interaction.

---

## **Tech Stack**

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **Embeddings** | Sentence-Transformers (BGE Small) |
| **Vector Indexing** | FAISS |
| **Language Model** | Gemini 2.0 Flash (Google Generative AI) |
| **Dataset** | Finance-Alpaca |
| **Language** | Python 3.10+ |

---

## **Installation & Setup**

### Clone the repository
```bash
git clone https://github.com/nuricanaksu/finance-rag-chatbot.git
cd finance-rag-chatbot
