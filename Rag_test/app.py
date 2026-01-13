import streamlit as st
import requests
from typing import List
import time

# Page config
st.set_page_config(
    page_title="RAG Knowledge Base",
    page_icon="📚",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"

def upload_pdfs(files: List) -> dict:
    """Upload PDFs to the backend"""
    files_data = [("files", (file.name, file, "application/pdf")) for file in files]
    response = requests.post(f"{API_URL}/upload-pdfs", files=files_data)
    response.raise_for_status()
    return response.json()

def query_knowledge_base(question: str, chat_history: List[dict], top_k: int = 3) -> dict:
    """Query the knowledge base with chat history"""
    response = requests.post(
        f"{API_URL}/query",
        json={
            "question": question, 
            "top_k": top_k,
            "chat_history": chat_history
        }
    )
    response.raise_for_status()
    return response.json()

def clear_database():
    """Clear the database"""
    response = requests.delete(f"{API_URL}/clear-database")
    response.raise_for_status()
    return response.json()

def check_health():
    """Check API health"""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.json()
    except:
        return None

def check_processing_status():
    """Check if PDFs are being processed"""
    try:
        response = requests.get(f"{API_URL}/processing-status")
        return response.json()
    except:
        return None

def format_chat_history_for_api(messages: List[dict]) -> List[dict]:
    """Format chat messages for API (exclude sources)"""
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
    ]

# Custom CSS
st.markdown("""
<style>
    section[data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 999 !important;
        background-color: rgb(14, 17, 23) !important;
        padding: 1rem 1rem 1rem 0 !important;
        border-top: 1px solid rgba(250, 250, 250, 0.1) !important;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.3) !important;
    }
    
    div[data-testid="stChatInputContainer"] {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 999 !important;
        background-color: rgb(14, 17, 23) !important;
        padding: 1rem 1rem 1rem 0 !important;
        border-top: 1px solid rgba(250, 250, 250, 0.1) !important;
    }
    
    .main .block-container {
        padding-bottom: 100px !important;
    }
    
    .element-container:has(> div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stChatMessage"])) {
        padding-bottom: 90px !important;
    }
    
    div[data-testid="stChatMessage"] {
        margin-bottom: 1rem !important;
    }
    
    section[data-testid="stChatInput"] input:disabled,
    div[data-testid="stChatInputContainer"] input:disabled {
        opacity: 0.5 !important;
        cursor: not-allowed !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# Main UI
st.title("📚 RAG Knowledge Base with Image Understanding & Web Search")
st.markdown("Ask questions about your documents - intelligent search combines PDFs and web sources")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Health check
    health = check_health()
    if health:
        st.success(f"✅ API Connected")
        st.info(f"Documents in DB: {health.get('collection_count', 0)}")
        
        # Show processing status
        if health.get('processing', False):
            st.warning("⏳ Processing PDFs...")
            processing = check_processing_status()
            if processing and processing.get('is_processing'):
                progress = processing.get('current', 0) / max(processing.get('total', 1), 1)
                st.progress(progress)
                st.caption(f"Processing: {processing.get('current')}/{processing.get('total')}")
                st.caption(f"Current: {processing.get('current_file', '')}")
    else:
        st.error("❌ API Not Connected")
        st.warning("Make sure FastAPI server is running on port 8000")
    
    st.divider()
    
    # Upload section (kept for manual uploads)
    st.header("📤 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to add to the knowledge base"
    )
    
    if st.button("Upload & Process", type="primary", disabled=not uploaded_files):
        with st.spinner("Processing PDFs (text + images)..."):
            try:
                result = upload_pdfs(uploaded_files)
                st.success(f"✅ {result.get('message', 'Success')}")
                if 'total_chunks' in result:
                    st.info(f"Total chunks: {result['total_chunks']}")
                st.rerun()
            except requests.exceptions.RequestException as e:
                error_detail = "Unknown error"
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_data = e.response.json()
                        error_detail = error_data.get('detail', str(e))
                    except:
                        error_detail = e.response.text or str(e)
                st.error(f"Error: {error_detail}")
    
    st.divider()
    
    # Advanced settings
    st.header("🔧 Advanced")
    top_k = st.slider("Number of sources to retrieve", 1, 10, 3)
    
    # Show conversation stats
    if "messages" in st.session_state and st.session_state.messages:
        msg_count = len(st.session_state.messages)
        st.metric("Messages in conversation", msg_count)
    
    # Clear conversation button
    if st.button("🔄 Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Clear database
    if st.button("🗑️ Clear Database", type="secondary"):
        if st.session_state.get('confirm_clear'):
            try:
                result = clear_database()
                st.success(result.get('message', 'Database cleared'))
                st.session_state.confirm_clear = False
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.session_state.confirm_clear = True
            st.warning("Click again to confirm")

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.header("💬 Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    # Display all messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📄 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")
    
    # Process new messages
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        has_response = len(st.session_state.messages) > 1 and st.session_state.messages[-2]["role"] == "assistant"
        
        if not has_response and not st.session_state.get('processing', False):
            st.session_state.processing = True
            user_prompt = st.session_state.messages[-1]["content"]
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        chat_history = format_chat_history_for_api(st.session_state.messages[:-1])
                        response = query_knowledge_base(user_prompt, chat_history, top_k)
                        
                        if 'answer' in response:
                            st.markdown(response["answer"])
                            
                            sources = response.get("sources", [])
                            if sources:
                                with st.expander("📄 Sources"):
                                    for source in sources:
                                        st.markdown(f"- {source}")
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response["answer"],
                                "sources": sources
                            })
                        else:
                            error_msg = "Error: Unexpected response format"
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_msg,
                                "sources": []
                            })
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "sources": []
                        })
                    finally:
                        st.session_state.processing = False
                        st.rerun()

with col2:
    st.header("ℹ️ Quick Info")
    st.markdown("""
    **Features:**
    • Text + Image understanding
    • Intelligent web search
    • Conversational memory
    • Source citations
    
    **Auto-Processing:**
    • PDFs in `/data/pdfs/` are auto-processed on startup
    • Only new/changed files are processed
    """)
    
    st.divider()
    
    st.markdown("**Example Questions:**")
    st.markdown("""
    • "What are the main topics?"
    • "Show me troubleshooting steps"
    • "How do I fix error X?"
    • "What does the screenshot on page 5 show?"
    """)

# Chat input
chat_disabled = st.session_state.get('processing', False)

if chat_disabled:
    st.info("⏳ Processing your question...")

if prompt := st.chat_input("Ask a question about your documents...", disabled=chat_disabled):
    if not chat_disabled:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>RAG with Image Understanding & Web Search | Powered by Ollama (LLaVA), ChromaDB, FastAPI & Streamlit</small>
</div>
""", unsafe_allow_html=True)