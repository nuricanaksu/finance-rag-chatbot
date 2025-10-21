# FinSage AI — Retrieval-Augmented Financial Chatbot

FinSage AI is an educational financial assistant designed to provide context-aware insights based on the Finance-Alpaca dataset.  
It integrates Google’s Gemini 2.0 Flash model with a FAISS vector database to deliver relevant and accurate financial information using Retrieval-Augmented Generation (RAG).

---

## Overview

The system retrieves the most relevant financial data chunks, constructs a context-rich prompt, and generates an answer through a generative model.  
It is built primarily for research, educational, and conceptual learning purposes — not for financial advice or live market analysis.

---

## Key Features

- RAG-based architecture combining FAISS and Gemini 2.0 Flash  
- Trained on the Finance-Alpaca dataset  
- Contextual question answering for finance-related queries  
- Interactive chat-style interface built with Streamlit  
- Modular design for easy extension and fine-tuning

---

## Architecture

| Component | Description |
|------------|-------------|
| LLM | Gemini 2.0 Flash (Google Generative AI) |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Database | FAISS |
| Dataset | Finance-Alpaca (Hugging Face) |
| UI Layer | Streamlit |
| Language | Python 3.10+ |

---

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/nuricanaksu/finance-rag-chatbot.git
cd finance-rag-chatbot
