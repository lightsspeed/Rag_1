import streamlit as st
import requests
from typing import List

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
    response.raise_for_status()  # Raise an exception for bad status codes
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
    response.raise_for_status()  # Raise an exception for bad status codes
    return response.json()

def clear_database():
    """Clear the database"""
    response = requests.delete(f"{API_URL}/clear-database")
    response.raise_for_status()  # Raise an exception for bad status codes
    return response.json()

def check_health():
    """Check API health"""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.json()
    except:
        return None

def format_chat_history_for_api(messages: List[dict]) -> List[dict]:
    """Format chat messages for API (exclude sources)"""
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
    ]

# Add custom CSS to fix chat input at bottom
st.markdown("""
<style>
    /* Fix chat input container at bottom */
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
    
    /* Fix chat input at bottom - alternative selector */
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
    
    /* Add padding to main content so it doesn't get hidden behind fixed input */
    .main .block-container {
        padding-bottom: 100px !important;
    }
    
    /* Add spacing to chat column so messages don't get cut off */
    .element-container:has(> div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stChatMessage"])) {
        padding-bottom: 90px !important;
    }
    
    /* Ensure proper spacing */
    div[data-testid="stChatMessage"] {
        margin-bottom: 1rem !important;
    }
    
    /* Style disabled chat input */
    section[data-testid="stChatInput"] input:disabled,
    div[data-testid="stChatInputContainer"] input:disabled {
        opacity: 0.5 !important;
        cursor: not-allowed !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Make spinner more visible */
    div[data-testid="stSpinner"] {
        margin: 1rem 0 !important;
    }
    
    /* Ensure spinner container is visible */
    div[data-testid="stSpinner"] > div {
        color: #ff4b4b !important;
    }
</style>
<script>
    // Auto-scroll to bottom when new messages are added
    function scrollToBottom() {
        // Scroll to show the chat input area (accounting for fixed position)
        const scrollHeight = Math.max(
            document.body.scrollHeight,
            document.documentElement.scrollHeight,
            document.body.offsetHeight,
            document.documentElement.offsetHeight,
            document.body.clientHeight,
            document.documentElement.clientHeight
        );
        window.scrollTo({
            top: scrollHeight - window.innerHeight,
            behavior: 'smooth'
        });
    }
    
    // Scroll immediately on load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(scrollToBottom, 300);
        });
    } else {
        setTimeout(scrollToBottom, 300);
    }
    
    // Use MutationObserver to detect when new chat messages are added
    let scrollTimeout;
    const observer = new MutationObserver(function(mutations) {
        // Debounce scrolling to avoid too many scrolls
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(function() {
            scrollToBottom();
        }, 100);
    });
    
    // Start observing when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            observer.observe(document.body, {
                childList: true,
                subtree: true,
                attributes: false
            });
        });
    } else {
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: false
        });
    }
    
    // Also scroll when spinner appears (thinking indicator)
    const spinnerObserver = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.addedNodes.length) {
                mutation.addedNodes.forEach(function(node) {
                    if (node.nodeType === 1 && (
                        node.querySelector && node.querySelector('[data-testid="stSpinner"]') ||
                        node.getAttribute && node.getAttribute('data-testid') === 'stSpinner'
                    )) {
                        setTimeout(scrollToBottom, 100);
                    }
                });
            }
        });
    });
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            spinnerObserver.observe(document.body, {
                childList: true,
                subtree: true
            });
        });
    } else {
        spinnerObserver.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
</script>
""", unsafe_allow_html=True)

# Main UI
st.title("📚 RAG Knowledge Base with Conversational Memory")
st.markdown("Upload PDFs and have natural conversations about their content")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Health check
    health = check_health()
    if health:
        st.success(f"✅ API Connected")
        st.info(f"Documents in DB: {health.get('collection_count', 0)}")
    else:
        st.error("❌ API Not Connected")
        st.warning("Make sure FastAPI server is running on port 8000")
    
    st.divider()
    
    # Upload section
    st.header("📤 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to add to the knowledge base"
    )
    
    if st.button("Upload & Process", type="primary", disabled=not uploaded_files):
        with st.spinner("Processing PDFs..."):
            try:
                result = upload_pdfs(uploaded_files)
                if 'message' in result:
                    st.success(f"✅ {result['message']}")
                else:
                    st.success("✅ PDFs processed successfully")
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
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
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
                if 'message' in result:
                    st.success(result['message'])
                else:
                    st.success("✅ Database cleared successfully")
                st.session_state.confirm_clear = False
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
    
    # Track if we need to process a new user message
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    # Display all existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📄 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")
    
    # Check if we need to process a new user message
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        # Check if we already have a response for this message
        # A response exists if there's an assistant message after the last user message
        has_response = len(st.session_state.messages) > 1 and st.session_state.messages[-2]["role"] == "assistant"
        
        # Process if: no response exists AND not currently processing
        if not has_response and not st.session_state.get('processing', False):
            # Set processing flag first to disable input
            st.session_state.processing = True
            # Continue processing immediately - no need for intermediate rerun
            user_prompt = st.session_state.messages[-1]["content"]
            
            # Show assistant message with spinner
            with st.chat_message("assistant"):
                # Show spinner with clear message
                with st.spinner("Thinking..."):
                    try:
                        # Format chat history for API (without sources field)
                        chat_history = format_chat_history_for_api(st.session_state.messages[:-1])
                        
                        # Make API call - this will block and show spinner
                        response = query_knowledge_base(user_prompt, chat_history, top_k)
                        
                        if 'answer' in response:
                            st.markdown(response["answer"])
                            
                            sources = response.get("sources", [])
                            if sources:
                                with st.expander("📄 Sources"):
                                    for source in sources:
                                        st.markdown(f"- {source}")
                            
                            # Add assistant message
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response["answer"],
                                "sources": sources
                            })
                        else:
                            error_msg = "Error: Unexpected response format from API"
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_msg,
                                "sources": []
                            })
                    except requests.exceptions.RequestException as e:
                        error_detail = "Unknown error"
                        if hasattr(e, 'response') and e.response is not None:
                            try:
                                error_data = e.response.json()
                                error_detail = error_data.get('detail', str(e))
                            except:
                                error_detail = e.response.text or str(e)
                        error_msg = f"Error: {error_detail}"
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
                        # Reset processing flag
                        st.session_state.processing = False
                        # Only rerun once after everything is complete
                        st.rerun()
    

with col2:
    st.header("ℹ️ Quick Info")
    st.markdown("""
    **Steps:**
    1. Upload PDFs (sidebar)
    2. Ask questions below
    
    **Features:**
    • Conversational memory
    • Context-aware responses
    • Source citations
    """)
    
    st.divider()
    
    st.markdown("**Example Questions:**")
    st.markdown("""
    • "What are the main topics?"
    • "Tell me more about X"
    • "Summarize chapter 3"
    """)

# Chat input fixed at bottom (outside columns so it spans full width)
# Disable input during processing
chat_disabled = st.session_state.get('processing', False)

# Show status if processing
if chat_disabled:
    st.info("⏳ Processing your question... Please wait.")

if prompt := st.chat_input("Ask a question about your documents...", disabled=chat_disabled):
    # Only process if not already processing
    if not chat_disabled:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Don't set processing flag here - let it be set in the processing section
        st.rerun()

# Footer (hidden behind fixed chat input, but kept for structure)
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>RAG Application with Conversational Memory | Powered by Ollama, ChromaDB, FastAPI & Streamlit</small>
</div>
""", unsafe_allow_html=True)