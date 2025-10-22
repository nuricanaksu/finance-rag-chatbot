"""
FinSage AI - Financial Insight Assistant with RAG Architecture

TECHNICAL OVERVIEW:
This application implements a Retrieval-Augmented Generation (RAG) system for financial Q&A.
It combines semantic search with generative AI to provide accurate, context-aware responses.

ARCHITECTURE COMPONENTS:
1. Data Layer: Finance-Alpaca dataset (52,000+ instruction-tuned financial examples)
2. Embedding Layer: BAAI/bge-small-en-v1.5 (384-dimensional embeddings)
3. Vector Store: FAISS with Inner Product similarity search
4. Generation Layer: Google Gemini 2.0 Flash for answer synthesis
5. UI Layer: Streamlit for interactive web interface

WORKFLOW:
User Query → Embedding → Vector Search → Context Retrieval → Prompt Construction → LLM Generation → Response
"""

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
# 1️⃣ CONFIGURATION & ENVIRONMENT SETUP
# ==============================
"""
ENVIRONMENT CONFIGURATION:
- Load API keys from .env file for security
- Configure Google Generative AI client
- Set up file paths for data persistence
"""

load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("GOOGLE_API_KEY")

# Validate API key presence - critical for LLM functionality
if not api_key:
    st.error("🚨 GOOGLE_API_KEY not found in .env file or environment variables.")
    st.stop()

# Initialize Google Generative AI with API key
genai.configure(api_key=api_key)

# MODEL SELECTION:
# - gemini-2.0-flash-exp: Fast inference, good for conversational AI
# - BAAI/bge-small-en-v1.5: Efficient 384-dim embeddings, optimized for semantic search
GEN_MODEL = "gemini-2.0-flash-exp"
EMB_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# FILE PATHS: Store processed data locally for faster subsequent runs
DATA_DIR = os.path.join(os.getcwd(), "data")
PASSAGES_PATH = os.path.join(DATA_DIR, "passages.jsonl")  # Serialized text passages
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")  # FAISS vector index

os.makedirs(DATA_DIR, exist_ok=True)  # Create data directory if doesn't exist

# ==============================
# 2️⃣ DATA PROCESSING & INDEXING
# ==============================
"""
DATA PIPELINE:
1. Download Finance-Alpaca dataset from HuggingFace
2. Format each example as: Instruction + Input + Answer
3. Generate embeddings using Sentence Transformers
4. Build FAISS index for efficient similarity search
5. Persist to disk for caching

CACHING STRATEGY:
- @st.cache_resource: Cache loaded models and indices across reruns
- Reduces cold start time from ~2 minutes to <1 second
"""

@st.cache_resource(show_spinner=False)
def get_embedder():
    """
    Load and cache the embedding model.
    
    MODEL: BAAI/bge-small-en-v1.5
    - 384 dimensions
    - Optimized for English text
    - Normalized embeddings for cosine similarity
    
    Returns:
        SentenceTransformer: Embedding model instance
    """
    return SentenceTransformer(EMB_MODEL_NAME)

@st.cache_resource(show_spinner=True)
def build_index_if_needed():
    """
    Build or load the FAISS index and passages.
    
    PROCESS:
    1. Check if preprocessed data exists
    2. If not, download Finance-Alpaca dataset
    3. Format passages as instruction-input-answer triplets
    4. Encode all passages to embeddings (batch processing)
    5. Create FAISS IndexFlatIP (Inner Product for cosine similarity)
    6. Save passages and index to disk
    
    FAISS INDEX TYPE:
    - IndexFlatIP: Exact search using inner product
    - Normalized embeddings → inner product = cosine similarity
    - Trade-off: Accuracy over speed (suitable for 50K vectors)
    
    Returns:
        tuple: (passages list, FAISS index)
    """
    # Load from cache if exists
    if os.path.exists(PASSAGES_PATH) and os.path.exists(INDEX_PATH):
        passages = [json.loads(l) for l in open(PASSAGES_PATH, "r", encoding="utf-8")]
        index = faiss.read_index(INDEX_PATH)
        return passages, index
    
    # Download dataset from HuggingFace
    st.info("📥 Downloading Finance-Alpaca dataset...")
    ds = load_dataset("gbharti/finance-alpaca", split="train")
    
    # Format passages: Combine instruction, input, and answer fields
    passages = []
    for r in ds:
        text = f"Instruction: {r.get('instruction','')}\nInput: {r.get('input','')}\nAnswer: {r.get('output','')}".strip()
        passages.append({"text": text})
    
    # Generate embeddings for all passages
    model = get_embedder()
    st.info("⚙️ Encoding embeddings...")
    embs = np.asarray(
        model.encode([p["text"] for p in passages], normalize_embeddings=True),
        dtype="float32"
    )
    
    # Build FAISS index
    # IndexFlatIP: Inner product similarity (equivalent to cosine for normalized vectors)
    index = faiss.IndexFlatIP(embs.shape[1])  # 384 dimensions
    index.add(embs)  # Add all embeddings to index
    
    # Persist to disk
    with open(PASSAGES_PATH, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    faiss.write_index(index, INDEX_PATH)
    return passages, index

@st.cache_resource
def load_assets():
    """
    Load preprocessed passages and FAISS index from disk.
    
    Returns:
        tuple: (passages list, FAISS index)
    """
    return [json.loads(l) for l in open(PASSAGES_PATH, "r", encoding="utf-8")], faiss.read_index(INDEX_PATH)

# ==============================
# 3️⃣ RAG PIPELINE
# ==============================
"""
RETRIEVAL-AUGMENTED GENERATION PIPELINE:

1. RETRIEVAL PHASE:
   - Encode user query to embedding vector
   - Search FAISS index for top-k similar passages
   - Return ranked contexts with similarity scores

2. PROMPT CONSTRUCTION:
   - Combine retrieved contexts into a single prompt
   - Add system instructions and constraints
   - Include user question

3. GENERATION PHASE:
   - Send prompt to Gemini LLM
   - Generate contextually grounded answer
   - Fallback to "not trained" response if context insufficient
"""

def retrieve(passages, index, query, k=5):
    """
    Retrieve top-k most similar passages for a query.
    
    ALGORITHM:
    1. Encode query using same embedding model
    2. Normalize query vector
    3. Search FAISS index using inner product
    4. Return top-k passages with similarity scores
    
    Args:
        passages (list): All passage texts
        index (faiss.Index): FAISS vector index
        query (str): User's question
        k (int): Number of passages to retrieve
    
    Returns:
        list: Top-k passages with scores, sorted by relevance
    """
    # Encode query to embedding vector
    q = get_embedder().encode([query], normalize_embeddings=True).astype("float32")
    
    # Search index: D = distances/scores, I = indices
    D, I = index.search(q, k)
    
    # Return passages with scores
    return [
        {"text": passages[idx]["text"], "score": float(D[0][rank])}
        for rank, idx in enumerate(I[0])
    ]

def build_prompt(question, contexts):
    """
    Construct the prompt for the LLM.
    
    PROMPT ENGINEERING:
    - System role: Define assistant's persona and constraints
    - Context injection: Provide retrieved passages as grounding
    - Question: User's original query
    - Fallback instruction: Handle out-of-scope queries gracefully
    
    Args:
        question (str): User's question
        contexts (list): Retrieved passages with scores
    
    Returns:
        str: Formatted prompt for LLM
    """
    # Format contexts with numbered references
    ctx = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)])
    
    # Construct prompt with system instructions
    return f"""
You are **FinSage AI**, an educational financial assistant trained exclusively on the Finance-Alpaca dataset.
Use only the provided context to answer. If the information is not found or unrelated, respond strictly with:
"I wasn't trained on that topic, so I can't provide a reliable answer."

Context:
{ctx}

Question: {question}
"""

def generate_answer(prompt: str) -> str:
    """
    Generate answer using Google Gemini LLM.
    
    ERROR HANDLING:
    - Try-catch for API failures
    - Fallback response for empty outputs
    - User-friendly error messages
    
    Args:
        prompt (str): Constructed prompt with context and question
    
    Returns:
        str: Generated answer or error message
    """
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel(GEN_MODEL)
        
        # Generate response
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        
        # Fallback for empty responses
        return text or "I wasn't trained on that topic, so I can't provide a reliable answer."
    except Exception as e:
        return f"⚠️ API Error: {e}"

# ==============================
# 4️⃣ STREAMLIT USER INTERFACE
# ==============================
"""
UI ARCHITECTURE:
- Dark theme with teal accent colors (#00E6A8)
- Two-column layout: Chat history (70%) + Example questions (30%)
- Custom CSS for chat bubbles, typing animation, and modal
- Session state management for conversation history
- Streaming response with word-by-word animation
"""

st.set_page_config(
    page_title="FinSage AI — Financial Insight Assistant",
    page_icon="💠",
    layout="wide",
)

# ==============================
# STYLING - PART 1: Core Layout
# ==============================
"""
CSS ARCHITECTURE:
- Base theme: Dark mode (#0D1117 background)
- Chat bubbles: Differentiated user/bot styles
- Animations: Typing indicator with CSS keyframes
- Responsive design: Max-width constraints for readability
"""

st.markdown("""
<style>
body, .stApp {
    background-color: #0D1117;
    color: #E8FDF5;
    font-family: 'Inter', sans-serif;
}
.chat-container {
    max-width: 950px;
    margin: auto;
    padding-bottom: 2rem;
}
.chat-bubble {
    padding: 1rem 1.2rem;
    border-radius: 0.75rem;
    margin-bottom: 1rem;
    max-width: 75%;
    font-size: 1rem;
    line-height: 1.6;
}
.user-msg {
    background-color: #00E6A8;
    color: #0D1117;
    margin-left: auto;
    text-align: right;
}
.bot-msg {
    background-color: #161B22;
    color: #E8FDF5;
    border: 1px solid rgba(0,230,168,0.2);
    box-shadow: 0 0 10px rgba(0,230,168,0.1);
}
.dot {
    height: 10px;
    width: 10px;
    margin: 0 3px;
    background-color: #00E6A8;
    border-radius: 50%;
    display: inline-block;
    animation: blink 1.4s infinite both;
}
@keyframes blink {
    0%,80%,100%{opacity:0;}
    40%{opacity:1;}
}
</style>
""", unsafe_allow_html=True)

# ==============================
# STYLING - PART 2: Components
# ==============================
"""
COMPONENT STYLING:
- Example questions: White cards with hover effects
- Buttons: Hover state transitions
- Modal: Fixed overlay with centered content
- Input field: Custom colors matching theme
"""

st.markdown("""
<style>
.example-box {
    background-color: #FFFFFF;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #0D1117;
}
.example-box h3 {
    color: #0D1117 !important;
    font-weight: 700;
    margin-bottom: 0.6rem;
}
div[data-testid="stButton"] > button {
    background-color: #FFFFFF !important;
    color: #0D1117 !important;
    border: 1px solid #CCCCCC !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    text-align: left !important;
    width: 100% !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: all 0.2s ease;
}
div[data-testid="stButton"] > button:hover {
    background-color: #00E6A8 !important;
    color: #FFFFFF !important;
    border-color: #00B383 !important;
    transform: translateY(-1px);
}
.info-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0,0,0,0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
}
.info-modal {
    background-color: #FFFFFF;
    color: #0D1117;
    padding: 2rem;
    border-radius: 12px;
    max-width: 600px;
    box-shadow: 0 0 20px rgba(0,0,0,0.3);
}
.info-close {
    background-color: #00E6A8;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.2rem;
    cursor: pointer;
    margin-top: 1rem;
    font-weight: 600;
}
.info-close:hover {
    background-color: #00C690;
}
.stChatInput textarea {
    background-color:#FFFFFF !important;
    color:#0D1117 !important;
    border-radius:10px !important;
    border:2px solid #00E6A8 !important;
}
.disclaimer {
    font-size:0.85rem;
    text-align:center;
    color:#A9B1B8;
    margin-top:1.5rem;
    border-top:1px solid rgba(255,255,255,0.1);
    padding-top:0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER SECTION
# ==============================
"""
HEADER LAYOUT:
- Three-column layout: Empty | Title | Info button
- Info button triggers modal with dataset information
"""

col1, col2, col3 = st.columns([0.08, 0.84, 0.08])
with col2:
    st.markdown("<h1 style='color:#00E6A8;margin-top:10px;'>FinSage AI — Your Financial Insight Partner</h1>", unsafe_allow_html=True)
with col3:
    if st.button("ℹ️", key="info_btn", help="View dataset info and limitations"):
        st.session_state["show_info"] = True

# ==============================
# INFORMATION MODAL
# ==============================
"""
MODAL FUNCTIONALITY:
- Displays dataset information and limitations
- Fixed overlay with centered modal
- Close button resets URL to clear modal state
"""

if st.session_state.get("show_info", False):
    st.markdown("""
    <div class='info-overlay'>
        <div class='info-modal'>
            <h3>📘 About FinSage AI</h3>
            <p><b>FinSage AI</b> is an educational assistant trained on the <b>Finance-Alpaca</b> dataset.</p>
            <ul>
                <li>Focuses on finance, banking, markets, and economic concepts.</li>
                <li>Trained using Stanford Alpaca's instruction format for finance education.</li>
                <li>Ideal for conceptual learning, not real-time advice.</li>
            </ul>
            <h3>⚠️ Limitations</h3>
            <ul>
                <li>No cryptocurrency or post-2023 financial data.</li>
                <li>No access to real-time markets or personal advice.</li>
                <li>Purely educational — do not use for investment decisions.</li>
            </ul>
            <br>
            <form action="/" method="get">
                <button class='info-close' type="submit">Close</button>
            </form>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==============================
# DATA INITIALIZATION
# ==============================
"""
INITIALIZATION LOGIC:
- Check if preprocessed data exists
- If not, trigger build_index_if_needed() with spinner
- Otherwise, load from cache for instant startup
"""

if not (os.path.exists(PASSAGES_PATH) and os.path.exists(INDEX_PATH)):
    with st.spinner("⏳ Preparing dataset..."):
        passages, index = build_index_if_needed()
else:
    passages, index = load_assets()

# ==============================
# CHAT INTERFACE
# ==============================
"""
CHAT STATE MANAGEMENT:
- Session state stores conversation history
- Messages format: {"role": "user/assistant", "content": "..."}
- Initial welcome message from assistant
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Welcome to FinSage AI. Ask a finance-related question."}
    ]

# Two-column layout: Chat history (70%) + Example questions (30%)
col_chat, col_examples = st.columns([0.7, 0.3])

# ==============================
# CHAT HISTORY DISPLAY
# ==============================
"""
CHAT RENDERING:
- Iterate through message history
- Apply different styles for user vs assistant messages
- Custom HTML for chat bubble design
"""

with col_chat:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state["messages"]:
        role = "bot-msg" if msg["role"] == "assistant" else "user-msg"
        st.markdown(f"<div class='chat-bubble {role}'>{msg['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# EXAMPLE QUESTIONS SIDEBAR
# ==============================
"""
EXAMPLE QUESTIONS:
- Pre-defined finance questions for quick testing
- Clicking a button sets selected_question in session state
- Triggers input processing in next section
"""

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

# ==============================
# INPUT PROCESSING & RESPONSE GENERATION
# ==============================
"""
INPUT HANDLING:
1. Get user input from chat_input or example button
2. Add user message to conversation history
3. Show typing indicator animation
4. Execute RAG pipeline: Retrieve → Build prompt → Generate
5. Stream response word-by-word for better UX
6. Add assistant response to history
7. Rerun to refresh UI

STREAMING ANIMATION:
- Split response into words
- Display incrementally with 0.03s delay
- Creates typing effect for natural conversation feel
"""

user_input = st.chat_input("Ask a financial question...")

# Check if user clicked an example question
if "selected_question" in st.session_state:
    user_input = st.session_state.pop("selected_question")

if user_input:
    # Add user message to history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Show typing indicator
    typing_placeholder = st.empty()
    typing_placeholder.markdown(
        "<div class='chat-bubble bot-msg'><span class='dot'></span><span class='dot'></span><span class='dot'></span></div>",
        unsafe_allow_html=True,
    )
    
    # Execute RAG pipeline
    ctx = retrieve(passages, index, user_input, k=5)  # Retrieve top 5 contexts
    prompt = build_prompt(user_input, ctx)  # Build prompt with contexts
    answer = generate_answer(prompt)  # Generate answer with LLM
    
    # Remove typing indicator
    typing_placeholder.empty()
    
    # Stream response word-by-word
    bot_placeholder = st.empty()
    displayed = ""
    for word in answer.split():
        displayed += word + " "
        bot_placeholder.markdown(f"<div class='chat-bubble bot-msg'>{displayed}</div>", unsafe_allow_html=True)
        time.sleep(0.03)  # 30ms delay per word
    
    # Add assistant response to history
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    
    # Rerun to refresh UI with new message
    st.rerun()

# ==============================
# DISCLAIMER FOOTER
# ==============================
"""
LEGAL DISCLAIMER:
- Educational use only
- Not financial advice
- User responsibility acknowledgment
"""

st.markdown("""
<div class="disclaimer">
    ⚠️ FinSage AI provides educational information only. It does not offer investment or financial advice.<br>
    By using FinSage AI, you acknowledge that you are responsible for your own financial decisions.
</div>
""", unsafe_allow_html=True)