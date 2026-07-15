import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import subprocess
import warnings
import base64
from pathlib import Path

# Suppress deprecation warnings from external libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Ensure root directory is in path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from src.config import AVAILABLE_LLMS, DEFAULT_LLM, EMBEDDING_MODEL
from src.logger import get_logger
from src.vectorstore import build_vectorstore
from src.chain import build_chain
from src.retriever import build_retriever
from src.agent import build_agent

log = get_logger(__name__)

# Function to get base64 representation of SVG icons
def get_base64_svg(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        log.error(f"Error reading icon {file_path}: {e}")
        return ""

# Encode icons for dynamic favicon swapping
dark_svg_b64 = get_base64_svg(ROOT_DIR / "icons" / "ollama-logo-dark.svg")
light_svg_b64 = get_base64_svg(ROOT_DIR / "icons" / "ollama-logo-light.svg")

# Configure Streamlit page options (light logo as default)
st.set_page_config(
    page_title="RAG Ollama & LangChain UI",
    page_icon=str(ROOT_DIR / "icons" / "ollama-logo-light.svg"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium and Modern look
st.markdown("""
<style>
    /* Main Title Styling */
    .main-title {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8F8F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
        text-align: left;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #A0AEC0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        text-align: left;
    }
    
    /* Status Cards Configuration */
    .status-card {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    /* Chat Customization */
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Sources / Documents Highlight */
    .source-container {
        background-color: rgba(0, 0, 0, 0.2);
        border-left: 3px solid #FF4B4B;
        padding: 10px;
        margin-top: 5px;
        margin-bottom: 5px;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }
    .source-header {
        font-weight: bold;
        color: #FF8F8F;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Inject dynamic theme-based favicon swapping using JS
if dark_svg_b64 and light_svg_b64:
    js_code = f"""
    <script>
        const darkIcon = "data:image/svg+xml;base64,{dark_svg_b64}";
        const lightIcon = "data:image/svg+xml;base64,{light_svg_b64}";
        
        function updateFavicon() {{
            try {{
                const isDarkMode = window.parent.matchMedia && window.parent.matchMedia('(prefers-color-scheme: dark)').matches;
                const iconUrl = isDarkMode ? darkIcon : lightIcon;
                
                const parentDoc = window.parent.document;
                let links = parentDoc.querySelectorAll("link[rel*='icon']");
                
                if (links.length === 0) {{
                    const newLink = parentDoc.createElement('link');
                    newLink.rel = 'icon';
                    newLink.href = iconUrl;
                    parentDoc.head.appendChild(newLink);
                }} else {{
                    links.forEach(link => {{
                        link.href = iconUrl;
                    }});
                }}
            }} catch (e) {{
                console.error("Error setting favicon:", e);
            }}
        }}
        
        // Execute initially and set event listener
        updateFavicon();
        if (window.parent.matchMedia) {{
            window.parent.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', updateFavicon);
        }}
    </script>
    """
    components.html(js_code, height=0, width=0)

# Helper function to generate responsive SVG HTML matching theme color
def get_responsive_svg_html(file_path, width="60px"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        if "<?xml" in svg_content:
            svg_content = svg_content[svg_content.find("<svg"):]
        # Use currentColor so it automatically matches text colors in Streamlit theme
        svg_content = svg_content.replace('style="fill:white;"', 'style="fill:currentColor;"')
        svg_content = svg_content.replace('fill="white"', 'fill="currentColor"')
        return f'<div style="width: {width}; color: var(--text-color);">{svg_content}</div>'
    except Exception as e:
        log.error(f"Error generating sidebar SVG: {e}")
        return ""

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False
    st.session_state.vectorstore = None
    st.session_state.chain = None
    st.session_state.agent = None
    st.session_state.active_model = None
    st.session_state.active_mode = None

# Render responsive sidebar logo
sidebar_logo_html = get_responsive_svg_html(ROOT_DIR / "icons" / "ollama-logo-dark.svg", width="60px")
if sidebar_logo_html:
    st.sidebar.markdown(sidebar_logo_html, unsafe_allow_html=True)
else:
    st.sidebar.image(str(ROOT_DIR / "icons" / "ollama-logo-light.svg"), width=60)

st.sidebar.markdown("<h2 style='margin-top: 10px;'>RAG Configuration</h2>", unsafe_allow_html=True)

# 1. RAG Engine selection (selectbox list instead of radio checks)
mode = st.sidebar.selectbox(
    "Select RAG Engine:",
    options=["Basic RAG Chain (main_chain)", "Agentic RAG (main_agent)"],
    index=0,
    help="Basic RAG: Simple linear flow. Agentic RAG: Stateful LangGraph agent with self-correction and search retries."
)

# 2. Ollama model selection (selectbox list)
model_choice = st.sidebar.selectbox(
    "Ollama Model (LLM):",
    options=AVAILABLE_LLMS,
    index=AVAILABLE_LLMS.index(DEFAULT_LLM) if DEFAULT_LLM in AVAILABLE_LLMS else 0,
    help="Available models installed locally via Ollama."
)

# 3. Force database retraining option
force_retrain = st.sidebar.checkbox(
    "Force Vector Database Retraining",
    value=False,
    help="If checked, rebuilds the vector database from files in the 'docs' folder."
)

# Apply settings button
init_button = st.sidebar.button("Apply Settings / Restart", use_container_width=True, type="primary")

# Maintenance section
st.sidebar.markdown("---")
st.sidebar.markdown("### Maintenance")
clear_chat = st.sidebar.button("Clear Chat History", use_container_width=True)
unload_models = st.sidebar.button("Free RAM (Stop Ollama)", use_container_width=True)

if clear_chat:
    st.session_state.messages = []
    st.toast("Chat history cleared!", icon="🧹")

if unload_models:
    with st.spinner("Stopping local model services..."):
        # Stop active model
        if st.session_state.active_model:
            subprocess.run(["ollama", "stop", st.session_state.active_model])
        # Stop embedding model
        subprocess.run(["ollama", "stop", EMBEDDING_MODEL])
        st.toast("Models unloaded from RAM!", icon="✅")

# Initialize RAG system function
def initialize_system(mode_selected: str, model_selected: str, retrain: bool):
    try:
        log.info(f"GUI: Initializing {mode_selected} with model={model_selected}, retrain={retrain}")
        
        # Build Vector Store
        vectorstore = build_vectorstore(force_retrain=retrain)
        st.session_state.vectorstore = vectorstore
        
        if "Basic RAG Chain" in mode_selected:
            # Build Chain
            chain = build_chain(vectorstore, model=model_selected)
            st.session_state.chain = chain
            st.session_state.agent = None
        else:
            # Build Agent
            retriever = build_retriever(vectorstore)
            agent = build_agent(retriever, model=model_selected)
            st.session_state.agent = agent
            st.session_state.chain = None
            
        st.session_state.active_model = model_selected
        st.session_state.active_mode = mode_selected
        st.session_state.system_initialized = True
        return True
    except Exception as e:
        st.error(f"Error initializing RAG: {e}")
        log.error(f"GUI initialization error: {e}")
        return False

# Trigger Auto-initialization or manual button click
if not st.session_state.system_initialized or init_button:
    with st.status("Initializing local RAG components...", expanded=True) as status:
        st.write("🔌 Connecting to Ollama...")
        st.write(f"📖 Checking document database (Force Retrain: {force_retrain})...")
        success = initialize_system(mode, model_choice, force_retrain)
        if success:
            status.update(label="RAG Ready for use!", state="complete", expanded=False)
            st.toast("RAG System ready!", icon="🚀")
        else:
            status.update(label="Initialization failed.", state="error", expanded=True)

# Main header
st.markdown("<div class='main-title'>Ollama RAG Playground</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle'>Interact with the local knowledge base using AI in {st.session_state.active_mode} mode</div>", unsafe_allow_html=True)

# Status indicators
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class='status-card'>
        <small style='color: #888;'>ACTIVE RAG ENGINE</small><br>
        <strong>{"Basic Chain (Linear)" if "Basic RAG Chain" in str(st.session_state.active_mode) else "Agentic Graph (LangGraph)"}</strong>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class='status-card'>
        <small style='color: #888;'>LANGUAGE MODEL (LLM)</small><br>
        <strong>{st.session_state.active_model}</strong>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class='status-card'>
        <small style='color: #888;'>EMBEDDING MODEL</small><br>
        <strong>{EMBEDDING_MODEL}</strong>
    </div>
    """, unsafe_allow_html=True)

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "documents" in message and message["documents"]:
            with st.expander("📚 Sources Consulted"):
                for idx, doc in enumerate(message["documents"]):
                    source_name = os.path.basename(doc.get("metadata", {}).get("source", "Unknown Document"))
                    st.markdown(f"""
                    <div class='source-container'>
                        <div class='source-header'>Source {idx+1}: {source_name}</div>
                        <div>{doc.get("page_content", "")}</div>
                    </div>
                    """, unsafe_allow_html=True)

# Process question input
if question := st.chat_input("Type your question..."):
    # Display user question
    with st.chat_message("user"):
        st.markdown(question)
    
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Process assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        sources_placeholder = st.container()
        
        answer = ""
        retrieved_docs = []
        
        try:
            if "Basic RAG Chain" in str(st.session_state.active_mode):
                # Basic Chain Mode
                with st.spinner("Retrieving sources and generating response..."):
                    # Retrieve sources (maintaining modularity)
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                    raw_docs = retriever.invoke(question)
                    retrieved_docs = [{"page_content": d.page_content, "metadata": d.metadata} for d in raw_docs]
                    
                    # Run linear chain
                    answer = st.session_state.chain.invoke(question)
                    response_placeholder.markdown(answer)
            else:
                # Agentic RAG Mode (LangGraph)
                with st.spinner("Agent analyzing and generating structured response..."):
                    result = st.session_state.agent.invoke({
                        "question":   question,
                        "documents":  [],
                        "generation": "",
                        "attempts":   0,
                    })
                    answer = result["generation"]
                    raw_docs = result.get("documents", [])
                    retrieved_docs = [{"page_content": d.page_content, "metadata": d.metadata} for d in raw_docs]
                    response_placeholder.markdown(answer)
            
            # Show sources if any
            if retrieved_docs:
                with sources_placeholder.expander("📚 Sources Consulted"):
                    for idx, doc in enumerate(retrieved_docs):
                        source_name = os.path.basename(doc.get("metadata", {}).get("source", "Unknown Document"))
                        st.markdown(f"""
                        <div class='source-container'>
                            <div class='source-header'>Source {idx+1}: {source_name}</div>
                            <div>{doc.get("page_content", "")}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Save message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "documents": retrieved_docs
            })
            
        except Exception as e:
            st.error(f"Error during question processing: {e}")
            log.error(f"Error processing question in GUI: {e}")
