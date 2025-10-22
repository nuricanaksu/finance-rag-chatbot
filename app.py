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

# ==============================
# 1️⃣ CONFIG & SETUP
# ==============================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("🚨 GOOGLE_API_KEY not found in .env file or environment variables.")
    st.stop()

genai.configure(api_key=api_key)
GEN_MODEL = "gemini-2.0-flash-exp"
EMB_MODEL_NAME = "BAAI/bge-small-en-v1.5"

DATA_DIR = os.path.join(os.getcwd(), "data")
PASSAGES_PATH = os.path.join(DATA_DIR, "passages.jsonl")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
os.makedirs(DATA_DIR, exist_ok=True)

# ==============================
# 2️⃣ DATA HANDLING
# ==============================
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMB_MODEL_NAME)

@st.cache_resource(show_spinner=True)
def build_index_if_needed():
    if os.path.exists(PASSAGES_PATH) and os.path.exists(INDEX_PATH):
        passages = [json.loads(l) for l in open(PASSAGES_PATH, "r", encoding="utf-8")]
        index = faiss.read_index(INDEX_PATH)
        return passages, index

    st.info("📥 Downloading Finance-Alpaca dataset...")
    ds = load_dataset("gbharti/finance-alpaca", split="train")

    passages = []
    for r in ds:
        text = f"Instruction: {r.get('instruction','')}\nInput: {r.get('input','')}\nAnswer: {r.get('output','')}".strip()
        passages.append({"text": text})

    model = get_embedder()
    st.info("⚙️ Encoding embeddings...")
    embs = np.asarray(model.encode([p["text"] for p in passages], normalize_embeddings=True), dtype="float32")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    with open(PASSAGES_PATH, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    faiss.write_index(index, INDEX_PATH)
    return passages, index

@st.cache_resource
def load_assets():
    return [json.loads(l) for l in open(PASSAGES_PATH, "r", encoding="utf-8")], faiss.read_index(INDEX_PATH)

# ==============================
# 3️⃣ RAG PIPELINE
# ==============================
def retrieve(passages, index, query, k=5):
    q = get_embedder().encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q, k)
    return [{"text": passages[idx]["text"], "score": float(D[0][rank])} for rank, idx in enumerate(I[0])]

def build_prompt(question, contexts):
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
    try:
        model = genai.GenerativeModel(GEN_MODEL)
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        return text or "I wasn’t trained on that topic, so I can’t provide a reliable answer."
    except Exception as e:
        return f"⚠️ API Error: {e}"

# ==============================
# 4️⃣ STREAMLIT UI
# ==============================
st.set_page_config(
    page_title="FinSage AI — Financial Insight Assistant",
    page_icon="💠",
    layout="wide",
)

# --- STYLING ---
st.markdown("""
<style>
body, .stApp { background-color: #0D1117; color: #E8FDF5; font-family: 'Inter', sans-serif; }
.chat-container { max-width: 950px; margin: auto; padding-bottom: 2rem; }
.chat-bubble { padding: 1rem 1.2rem; border-radius: 0.75rem; margin-bottom: 1rem; max-width: 75%; font-size: 1rem; line-height: 1.6; }
.user-msg { background-color: #00E6A8; color: #0D1117; margin-left: auto; text-align: right; }
.bot-msg { background-color: #161B22; color: #E8FDF5; border: 1px solid rgba(0,230,168,0.2); box-shadow: 0 0 10px rgba(0,230,168,0.1); }
.dot { height: 10px; width: 10px; margin: 0 3px; background-color: #00E6A8; border-radius: 50%; display: inline-block; animation: blink 1.4s infinite both; }
@keyframes blink { 0%,80%,100%{opacity:0;} 40%{opacity:1;} }

/* Example Questions */
.example-box { background-color: #FFFFFF; border-radius: 12px; padding: 1rem 1.2rem; color: #0D1117; }
.example-box h3 { color: #0D1117 !important; font-weight: 700; margin-bottom: 0.6rem; }
div[data-testid="stButton"] > button {
 background-color: #FFFFFF !important; color: #0D1117 !important;
 border: 1px solid #CCCCCC !important; border-radius: 8px !important;
 font-weight: 600 !important; text-align: left !important; width: 100% !important;
 box-shadow: 0 1px 4px rgba(0,0,0,0.05); transition: all 0.2s ease;
}
div[data-testid="stButton"] > button:hover {
 background-color: #00E6A8 !important; color: #FFFFFF !important;
 border-color: #00B383 !important; transform: translateY(-1px);
}

.stChatInput textarea { background-color:#FFFFFF !important; color:#0D1117 !important;
 border-radius:10px !important; border:2px solid #00E6A8 !important; }

.disclaimer { font-size:0.85rem; text-align:center; color:#A9B1B8;
 margin-top:1.5rem; border-top:1px solid rgba(255,255,255,0.1); padding-top:0.8rem; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col1, col2, col3 = st.columns([0.08, 0.84, 0.08])
with col2:
    st.markdown("<h1 style='color:#00E6A8;margin-top:10px;'>FinSage AI — Your Financial Insight Partner</h1>", unsafe_allow_html=True)
with col3:
    if st.button("ℹ️", key="info_btn", help="View dataset info and limitations"):
        st.session_state["show_info"] = True

# --- INFO MODAL ---
if st.session_state.get("show_info", False):
    st.markdown("""
    <div style='position:fixed;top:0;left:0;width:100%;height:100%;background-color:rgba(0,0,0,0.8);
    display:flex;align-items:center;justify-content:center;z-index:9999;'>
        <div style='background-color:#FFFFFF;color:#0D1117;padding:2rem;border-radius:12px;max-width:600px;'>
            <h3>📘 About FinSage AI</h3>
            <p><b>FinSage AI</b> is an educational assistant trained on the <b>Finance-Alpaca</b> dataset.</p>
            <ul>
                <li>Focuses on finance, banking, markets, and economic concepts.</li>
                <li>Ideal for conceptual learning, not real-time advice.</li>
            </ul>
            <h3>⚠️ Limitations</h3>
            <ul>
                <li>No cryptocurrency or post-2023 financial data.</li>
                <li>No access to real-time markets or personal advice.</li>
            </ul>
            <form action="/" method="get">
                <button style='background-color:#00E6A8;color:white;border:none;border-radius:6px;
                padding:0.5rem 1.2rem;cursor:pointer;margin-top:1rem;font-weight:600;'>Close</button>
            </form>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- LOAD DATA ---
if not (os.path.exists(PASSAGES_PATH) and os.path.exists(INDEX_PATH)):
    with st.spinner("⏳ Preparing dataset..."):
        passages, index = build_index_if_needed()
else:
    passages, index = load_assets()

# --- CHAT LAYOUT ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Welcome to FinSage AI. Ask a finance-related question."}]

col_chat, col_examples = st.columns([0.7, 0.3])
with col_chat:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state["messages"]:
        role = "bot-msg" if msg["role"] == "assistant" else "user-msg"
        st.markdown(f"<div class='chat-bubble {role}'>{msg['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_examples:
    st.markdown("<div class='example-box'><h3>💬 Example Questions</h3>", unsafe_allow_html=True)
    examples = [
        "What is the difference between assets and liabilities?",
        "Explain compound interest in simple terms.",
        "What does diversification mean in investing?",
        "How does inflation affect purchasing power?",
        "What is a balance sheet used for?",
        "What does liquidity mean in finance?",
        "Explain risk vs. return.",
        "What are the differences between stocks and bonds?",
        "How can an investor manage financial risk?",
        "What is the role of a central bank?"
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["selected_question"] = ex
    st.markdown("</div>", unsafe_allow_html=True)

# --- CHAT INPUT ---
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
⚠️ FinSage AI provides educational information only. It does not offer investment or financial advice.<br>
By using FinSage AI, you acknowledge that you are responsible for your own financial decisions.
</div>
""", unsafe_allow_html=True)
