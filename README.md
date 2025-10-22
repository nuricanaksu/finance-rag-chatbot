#  FinSage AI — Financial Insight Assistant *A Retrieval-Augmented Educational Finance Chatbot powered by Gemini 2.0 Flash* [![Streamlit App](https://img.shields.io/badge/🚀_Live_App-Streamlit-brightgreen?logo=streamlit)](https://finsageai.streamlit.app/) [![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/) [![Model](https://img.shields.io/badge/Model-Gemini_2.0_Flash-ff69b4?logo=google)](https://ai.google.dev/gemini-api) [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


Overview

FinSage AI is a Retrieval-Augmented Generation (RAG)–based educational chatbot that helps users understand key concepts in finance and economics.
It integrates semantic retrieval, vector search, and Google Gemini 2.0 Flash to provide grounded, context-aware explanations using the Finance-Alpaca dataset.

Tech Stack
Component	Description
Frontend	Streamlit UI with a custom dark theme and chat interface
Backend	Python-based RAG pipeline using FAISS and Gemini 2.0 Flash
LLM	Google Gemini 2.0 Flash
Embeddings	BAAI/bge-small-en-v1.5
Vector Store	FAISS (Flat Inner Product Index)
Dataset	gbharti/finance-alpaca

Deployment	Streamlit Cloud
Key Features

✅ Retrieval-Augmented Generation (RAG) pipeline

✅ Context-aware financial question answering

✅ Smooth typing animation for a natural chat experience

✅ Informational modal detailing dataset & limitations

✅ Pre-loaded example queries

✅ Secure API-key handling for local or cloud deployment

⚙️ Installation & Local Setup
1️⃣ Clone the repository
git clone https://github.com/nuricanaksu/finance-rag-chatbot.git
cd finance-rag-chatbot

2️⃣ Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# or
.venv\Scripts\activate         # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add your Google API key (securely)

Create a .env file in the project root:

echo "GOOGLE_API_KEY=your_api_key_here" > .env

5️⃣ Run the Streamlit app
streamlit run app.py


Then open http://localhost:8501
 in your browser.

🔐 API Key Security

Your API key must never be committed to GitHub.

To keep it safe:

Store it locally in .env, or

Use Streamlit Cloud Secrets Manager:

.streamlit/secrets.toml

GOOGLE_API_KEY = "your_api_key_here"

📘 Notebook Integration

This project includes Jupyter Notebooks for documentation and evaluation.

Section	Description
Section 1	RAG concepts, Gemini setup, data preprocessing
Section 2	Embedding generation, FAISS indexing, chatbot integration

These notebooks comply with the guideline:

“All technical explanations must appear in Markdown cells or inline comments.”

🎥 Demo Video

🎬 Coming soon!
A full walkthrough showing dataset loading, FAISS retrieval, and Gemini-powered response generation.
(Replace this line with your video link once uploaded.)

⚠️ Limitations

📚 For educational purposes only — not real-time or personalized financial advice

🧾 Dataset is static (pre-2023)

💹 No live market or cryptocurrency data access

📜 License

This project is released under the MIT License.
You are free to use, modify, and distribute it — with proper attribution.

<p align="center"> <sub>💡 Built with passion by <a href="https://github.com/nuricanaksu">Nuri Aksu</a> — Powered by <strong>Gemini 2.0 Flash</strong></sub> </p>