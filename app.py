"""
💠 FinSage AI — Financial Insight Assistant
A Retrieval-Augmented Generation (RAG) chatbot powered by Google Gemini 2.0 Flash.

This script implements a full RAG pipeline using:
- Dataset: Finance-Alpaca (Hugging Face)
- Embeddings: BAAI/bge-small-en-v1.5
- Vector Store: FAISS
- LLM: Gemini 2.0 Flash
- Frontend: Streamlit (custom styled chat interface)

Author: Nuri Aksu
"""

# ==============================
# 1️⃣ CONFIG & SETUP
# ==============================

import os
import json
import time
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# --- Load API key from environment or Streamlit Secrets ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("🚨 GOOGLE_API_KEY not found in .env file or environment variables.")
    st.stop()

# --- Configure Gemini model ---
genai.configure(api_key=api_key)
GEN_MODEL = "gemini-2.0-flash-exp"  # Google's latest lightweight, fast reasoning model
EMB_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Compact, high-quality sentence embedding model

# --- Data directory and file paths ---
DATA_DIR = os.path.join(os.getcwd(), "data")
PASSAGES_PATH = os.path.join(DATA_DIR, "passages.jsonl")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
os.makedirs(DATA_DIR, exist_ok=True)


# ==============================
# 2️⃣ DATA HANDLING (LOAD & INDEX)
# ==============================
# Here we prepare the dataset and vector index for retrieval.

@st.cache_resource(show_spinner=False)
def get_embedder():
    """Loads the sentence-transformer model only once to save memory."""
    return SentenceTransformer(EMB_MODEL_NAME)


@st.cache_resource(show_spinner=True)
def build_index_if_needed():
    """
    Builds the FAISS index for semantic retrieval.
    If already built, loads it from disk for faster startup.
    """
    # --- Check for existing index ---
    if os.path.exists(PASSAGES_PATH) and os.path.exists(INDEX_PATH):
        passages = [json.loads(l) for l in open(PASSAGES_PATH, "r", encoding="utf-8")]
        index = faiss.read_index(INDEX_PATH)
        return passages, index

    # --- Download dataset from Hugging Face ---
    st.info("📥 Downloading Finance-Alpaca dataset...")
    ds = load_dataset("gbharti/finance-alpaca", split="train")

    # --- Format data for embedding ---
    passages = []
    for r in ds:
        text = f"Instruction: {r.get('instruction','')}\nInput: {r.get('input','')}\nAnswer: {r.get('output','')}".strip()
        passages.append({"text": text})

    # --- Encode embeddings ---
    model = get_embedder()
    st.info("⚙️ Encoding embeddings using BAAI/bge-small-en-v1.5...")
    embs = np.asarray(model.encode([p["text"] for p in passages], normalize_embeddings=True), dtype="float32")

    # --- Build FAISS index (Inner Product for cosine similarity) ---
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    # --- Save to disk for reuse ---
    with open(PASSAGES_PATH, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    faiss.write_index(index, INDEX_PATH)
    return passages, index


@st.cache_resource
def load_assets():
    """Loads pre-built passages and FAISS index from local storage."""
    return [json.loads(l) for l in open(PASSAGES_PATH, "r", encoding="utf-8")], faiss.read_index(INDEX_PATH)


# ==============================
# 3️⃣ RAG PIPELINE (RETRIEVE + GENERATE)
# ==============================
# This section handles information retrieval and LLM response generation.

def retrieve(passages, index, query, k=5):
    """
    Retrieves the top-k semantically similar chunks from the FAISS index.
    """
    q = get_embedder().encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q, k)
    return [{"text": passages[idx]["text"], "score": float(D[0][rank])} for rank, idx in enumerate(I[0])]


def build_prompt(question, contexts):
    """
    Constructs a context-grounded prompt for Gemini 2.0 Flash.
    Includes retrieved passages for RAG-based response generation.
    """
    ctx = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)])
    return f"""
You are **FinSage AI**, an educational financial assistant trained exclusively on the Finance-Alpaca dataset.
Use only the provided context to answer.
If the information is not found or unrelated, respond strictly with:
"I wasn’t trained on that topic, so I can’t provide a reliable answer."

Context:
{ctx}

Question: {question}
"""


def generate_answer(prompt: str) -> str:
    """
    Sends the constructed prompt to Gemini 2.0 Flash and returns the model's response.
    """
    try:
        model = genai.GenerativeModel(GEN_MODEL)
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        return text or "I wasn’t trained on that topic, so I can’t provide a reliable answer."
    except Exception as e:
        return f"⚠️ API Error: {e}"


# ==============================
# 4️⃣ STREAMLIT UI & FRONTEND
# ==============================
# Custom-styled chat interface with example prompts and typing animation.

st.set_page_config(
    page_title="FinSage AI — Financial Insight Assistant",
    page_icon="💠",
    layout="wide",
)

# --- Custom CSS Styling ---
st.markdown("""[...CSS CODE SAME AS BEFORE...]""", unsafe_allow_html=True)


# --- HEADER ---
col1, col2, col3 = st.columns([0.08, 0.84, 0.08])

with col2:
    st.markdown(f"""
    <div style='display: flex; align-items: center; justify-content: center; gap: 12px; margin-top: 10px;'>
        <img src='https://raw.githubusercontent.com/nuricanaksu/finance-rag-chatbot/main/owl-logo.png' 
             width='55' style='filter: drop-shadow(0px 0px 4px rgba(0,230,168,0.4));'>
        <h1 style='color:#00E6A8; font-weight:700; margin:0;'>FinSage AI — Your Financial Insight Partner</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:
    if st.button("ℹ️", key="info_btn", help="View dataset info and limitations"):
        st.session_state["show_info"] = True



# --- INFO MODAL ---
if st.session_state.get("show_info", False):
    st.markdown("""[...MODAL CODE SAME AS BEFORE...]""", unsafe_allow_html=True)


# --- LOAD DATASET & INDEX ---
if not (os.path.exists(PASSAGES_PATH) and os.path.exists(INDEX_PATH)):
    with st.spinner("⏳ Preparing dataset..."):
        passages, index = build_index_if_needed()
else:
    passages, index = load_assets()


# --- CHAT HISTORY HANDLER ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Welcome to FinSage AI. Ask a finance-related question."}]


# --- CHAT LAYOUT ---
col_chat, col_examples = st.columns([0.7, 0.3])
with col_chat:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state["messages"]:
        role = "bot-msg" if msg["role"] == "assistant" else "user-msg"
        st.markdown(f"<div class='chat-bubble {role}'>{msg['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- EXAMPLE QUESTIONS ---
with col_examples:
    st.markdown("<div class='example-box'><h3>💬 Example Questions</h3>", unsafe_allow_html=True)
    examples = [
        "What is the difference between assets and liabilities?",
        "Explain compound interest in simple terms.",
        "What does diversification mean in investing?",
        "How does inflation affect purchasing power?",
        "What is a balance sheet used for?",
        "What does liquidity mean in finance?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["selected_question"] = ex
    st.markdown("</div>", unsafe_allow_html=True)


# --- CHAT INPUT & RESPONSE GENERATION ---
user_input = st.chat_input("Ask a financial question...")
if "selected_question" in st.session_state:
    user_input = st.session_state.pop("selected_question")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    typing_placeholder = st.empty()
    typing_placeholder.markdown(
        "<div class='chat-bubble bot-msg'><span class='dot'></span><span class='dot'></span><span class='dot'></span></div>",
        unsafe_allow_html=True,
    )

    ctx = retrieve(passages, index, user_input, k=5)
    prompt = build_prompt(user_input, ctx)
    answer = generate_answer(prompt)

    typing_placeholder.empty()
    bot_placeholder = st.empty()
    displayed = ""
    for word in answer.split():
        displayed += word + " "
        bot_placeholder.markdown(f"<div class='chat-bubble bot-msg'>{displayed}</div>", unsafe_allow_html=True)
        time.sleep(0.03)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.rerun()


# --- DISCLAIMER ---
st.markdown("""
<div class="disclaimer">
⚠️ FinSage AI provides educational information only.<br>
It does not offer investment or financial advice.<br>
By using FinSage AI, you acknowledge that you are responsible for your own financial decisions.
</div>
""", unsafe_allow_html=True)
