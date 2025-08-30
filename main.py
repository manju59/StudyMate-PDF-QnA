import streamlit as st
import fitz  # PyMuMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss  # FAISS for vector search
import re
import os
import time
import google.generativeai as genai

# --- Page Configuration and Custom CSS ---
st.set_page_config(
    page_title="StudyMate - PDF Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #f0f2f6; 
        color: #2c3e50;
    }
    .sidebar .sidebar-content {
        background: #34495e;
        color: white;
    }
    .stButton>button {
        background: #3498db;
        color: white;
        border-radius: 12px;
        padding: 12px 28px;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 6px 10px rgba(0,0,0,0.2);
    }
    .stTextInput>div>div>input {
        border-radius: 12px;
        padding: 14px;
        border: 2px solid #e0e0e0;
        background-color: white;
        transition: border 0.3s ease;
        color: var(--input-text-color); /* Use a CSS variable for dynamic color */
    }
    .stTextInput>div>div>input:focus {
        border: 2px solid #3498db;
        box-shadow: 0 0 0 0.25rem rgba(52, 152, 219, 0.25);
    }
    .answer-box {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
        border-left: 6px solid #3498db;
        color: #2c3e50; /* Explicitly set a dark color for readability */
    }
    .source-box {
        background: #ecf0f1;
        padding: 18px;
        border-radius: 12px;
        border-left: 4px solid #2c3e50;
        margin-bottom: 15px;
        transition: transform 0.2s ease;
        color: #2c3e50; /* Explicitly set a dark color for readability */
    }
    .source-box:hover {
        transform: translateX(5px);
    }
    .header {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 10px;
        font-weight: 700;
    }
    .subheader {
        color: #3498db;
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
        font-weight: 600;
    }
    .success-box {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 25px;
        border-left: 5px solid #43a047;
    }
    .info-box {
        background: #e3f2fd;
        color: #1565c0;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 20px;
        border-left: 5px solid #1976d2;
    }
    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 5px;
        color: #2c3e50; /* Explicitly set a dark color for readability */
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Management ---
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = False
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'file_names' not in st.session_state:
    st.session_state.file_names = []
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''


# --- Helper Functions ---
def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF file with page breaks"""
    try:
        pdf_data = uploaded_file.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        text = ""
        for page_num, page in enumerate(doc):
            text += page.get_text("text") + f"\n\n---PAGE {page_num + 1} OF {uploaded_file.name}---\n\n"
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def preprocess_text(text):
    """Clean and preprocess text for chunking"""
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text_by_words(text, chunk_size=400, overlap=80):
    """Splits text into chunks of a fixed word count with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def load_embedding_model():
    """Load the sentence transformer model once."""
    if st.session_state.embedding_model is None:
        with st.spinner("🔧 Loading embedding model..."):
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                st.session_state.embedding_model = model
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return None
    return st.session_state.embedding_model

def create_faiss_index(chunks):
    """Creates a FAISS index from text chunks."""
    model = load_embedding_model()
    if model is None:
        return None, None
    
    with st.spinner("🧠 Creating semantic embeddings & FAISS index..."):
        embeddings = model.encode(chunks, show_progress_bar=False, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()
        
        # FAISS index creation
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index, embeddings

def semantic_search_faiss(question, index, chunks, top_k=5):
    """Finds the most relevant text chunks using FAISS."""
    model = load_embedding_model()
    if model is None or index is None:
        return []
        
    question_embedding = model.encode([question], convert_to_tensor=True).cpu().numpy()
    
    # Use FAISS to find the top k nearest neighbors
    D, I = index.search(question_embedding, top_k)
    
    relevant_chunks = []
    for i in range(len(I[0])):
        idx = I[0][i]
        if idx != -1: 
            relevant_chunks.append(chunks[idx])
    
    return relevant_chunks

def generate_answer_with_gemini(question, relevant_chunks):
    """
    Uses the Google Gemini API to generate an accurate answer based on context.
    """
    if not st.session_state.api_key:
        return "Please enter your Gemini API key in the sidebar to get an AI-powered answer."
    
    try:
        genai.configure(api_key=st.session_state.api_key)
        
        if st.session_state.gemini_model is None:
            st.session_state.gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

        if not relevant_chunks:
            return "I couldn't find specific information in your documents to answer that question. Please try rephrasing or check your uploaded files."
    
        combined_context = " ".join(relevant_chunks)
    
        # Create a detailed prompt for the LLM
        prompt = f"""
        You are an expert Q&A system. Your task is to answer the user's question as accurately as possible based *only* on the provided context. 
        If the answer is not contained within the context, state that you cannot answer the question.
        
        Context:
        {combined_context}
        
        Question: {question}
        
        Answer:
        """
        
        response = st.session_state.gemini_model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        st.error(f"Error generating AI answer: {str(e)}")
        return "An error occurred while trying to generate an answer. Please check your API key."


# --- Streamlit App Layout ---
st.markdown("<h1 class='header'>📚 StudyMate: PDF Q&A System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Get accurate, contextual answers from your study materials.</p>", unsafe_allow_html=True)

# --- Sidebar for PDF upload and settings ---
with st.sidebar:
    st.markdown("<h2 style='color: white;'>📁 Upload Documents</h2>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Select PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("<h3 style='color: white;'>⚙ Settings</h3>", unsafe_allow_html=True)
    
    # Text input color selector
    text_color = st.color_picker('Choose input text color', '#2c3e50')
    st.markdown(f"""
    <style>
        :root {{
            --input-text-color: {text_color};
        }}
    </style>
    """, unsafe_allow_html=True)

    # API key input
    st.markdown("<h4 style='color: white;'>🔑 Gemini API Key</h4>", unsafe_allow_html=True)
    st.session_state.api_key = st.text_input("Enter your API Key", type="password", key="api_key_input", help="Get your key from Google AI Studio")

    chunk_size = st.slider("Context Chunk Size (words)", 200, 600, 400, help="Controls the size of text chunks for processing.")
    top_k = st.slider("Max Source Passages", 1, 7, 3, help="Number of most relevant passages to consider for the answer.")
    
    # Buttons
    process_button = st.button("🚀 Process Documents", use_container_width=True)
    clear_button = st.button("🔄 Clear All Documents", use_container_width=True)
    
    if clear_button:
        st.session_state.processed_docs = False
        st.session_state.text_chunks = []
        st.session_state.faiss_index = None
        st.session_state.embeddings = None
        st.session_state.file_names = []
        st.session_state.gemini_model = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style='color: #bdc3c7; font-size: small;'>
    <p><strong>Tips for Students:</strong></p>
    <ul>
        <li>Use clear, specific questions.</li>
        <li>The more documents you upload, the longer processing will take.</li>
        <li>Check the source passages to verify the answer.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# --- Main Content Area ---
if process_button and uploaded_files:
    with st.spinner("Processing your documents..."):
        progress_bar = st.progress(0)
        
        all_text = ""
        st.session_state.file_names = [f.name for f in uploaded_files]
        
        # Step 1: Extract and preprocess text
        for i, uploaded_file in enumerate(uploaded_files):
            text = extract_text_from_pdf(uploaded_file)
            if text:
                all_text += preprocess_text(text)
            progress_bar.progress((i + 1) / len(uploaded_files) * 0.3)

        if not all_text.strip():
            st.error("No text could be extracted from the PDFs. Please try different files.")
        else:
            # Step 2: Chunk text
            st.session_state.text_chunks = chunk_text_by_words(all_text, chunk_size=chunk_size)
            progress_bar.progress(0.5)

            # Step 3: Create FAISS index
            index, embeddings = create_faiss_index(st.session_state.text_chunks)
            progress_bar.progress(1.0)

            if index is not None:
                st.session_state.faiss_index = index
                st.session_state.embeddings = embeddings
                st.session_state.processed_docs = True
                
                # Display success metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-box"><h4>📄 Docs</h4><h3>{len(uploaded_files)}</h3></div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-box"><h4>🧩 Chunks</h4><h3>{len(st.session_state.text_chunks)}</h3></div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-box"><h4>🔍 Search</h4><h3>FAISS</h3></div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="success-box">
                    <h4>✅ Processing Complete!</h4>
                    <p>Your documents have been processed and indexed with FAISS for fast Q&A. You can now ask questions!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Failed to create the search index. The embedding model may have failed to load.")

if not st.session_state.processed_docs:
    st.markdown("""
    <div class="info-box">
        <h4>📚 How to Use StudyMate</h4>
        <ol>
            <li>Upload your PDF study materials using the sidebar.</li>
            <li>Enter your Gemini API key in the sidebar.</li>
            <li>Click "Process Documents" to prepare them for Q&A.</li>
            <li>Type your question and click "Ask Question".</li>
            <li>The system will provide a synthesized answer and show you the source.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
else:
    if st.session_state.file_names:
        with st.expander("📋 Processed Documents"):
            for name in st.session_state.file_names:
                st.write(f"• {name}")
    
    st.markdown("<h3 class='subheader'>Ask a Question</h3>", unsafe_allow_html=True)
    question = st.text_input("Enter your question:", placeholder="e.g., What are the key stages of photosynthesis?", label_visibility="collapsed")
    ask_button = st.button("🔍 Ask Question", use_container_width=True)
    
    if question and ask_button:
        with st.spinner("Finding the answer..."):
            # Step 4: Semantic Search
            relevant_chunks = semantic_search_faiss(
                question, 
                st.session_state.faiss_index, 
                st.session_state.text_chunks, 
                top_k=top_k
            )
            
            # Step 5: Answer Generation using Gemini API
            answer = generate_answer_with_gemini(question, relevant_chunks)
            
            # Display answer
            st.markdown("<h3 class='subheader'>Answer</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
            
            # Display sources
            if relevant_chunks:
                st.markdown("<h3 class='subheader'>Source Passages</h3>", unsafe_allow_html=True)
                for i, chunk in enumerate(relevant_chunks):
                    st.markdown(f"""
                    <div class="source-box">
                        <h4>📝 Passage {i+1}</h4>
                        <p>{chunk}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.info("This answer was generated based on the source passages above. Always cross-reference with your original documents.")


# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p><strong>StudyMate</strong> - Advanced PDF Q&A System for Students</p>
        <p>Powered by semantic search and natural language processing</p>
    </div>
    """, 
    unsafe_allow_html=True
)
