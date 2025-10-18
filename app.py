import streamlit as st
import os
import tempfile
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from groq import Groq
import pickle
import uuid
load_dotenv()
# App title and description
st.set_page_config(page_title="BookScribe AI", layout="wide")
st.title("📚 BookScribe AI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Use the keys
groq_client = Groq(api_key=GROQ_API_KEY)
st.markdown("""
Transform your PDFs into interactive knowledge bases with personalized summaries.
Upload a document, choose your learning style, and start exploring!
""")

# Initialize session state variables if they don't exist
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'current_vector_store' not in st.session_state:
    st.session_state.current_vector_store = None
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None
if 'chapter_summaries' not in st.session_state:
    st.session_state.chapter_summaries = {}

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Learning style selector
    learning_style = st.selectbox(
        "Select your learning style:",
        ["Visual learner", "Auditory learner", "Reading/writing learner", "Kinesthetic learner"]
    )
    
    # Choose LLM model
    llm_model = st.selectbox(
        "Select Groq LLM Model:",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
    )
    
    # Language model parameters
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    max_tokens = st.slider("Max Tokens", min_value=100, max_value=4000, value=1000, step=100)

# Initialize the embedding model (Hugging Face)
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Process the uploaded PDF
def process_pdf(pdf_file, file_name):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name
    
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Extract text from documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Group chunks into logical "chapters" - simplified approach
    chapters = []
    current_chapter = []
    current_page = None
    
    for chunk in chunks:
        page = chunk.metadata.get('page', 0)
        if current_page is None:
            current_page = page
        
        # Simple heuristic: new page could be new chapter
        if page != current_page and current_chapter:
            chapters.append(current_chapter)
            current_chapter = []
        
        current_chapter.append(chunk)
        current_page = page
    
    # Add the last chapter
    if current_chapter:
        chapters.append(current_chapter)
    
    # Create vector store with embeddings
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save vector store temporarily
    storage_path = f"temp_storage/{st.session_state.user_id}"
    os.makedirs(storage_path, exist_ok=True)
    
    with open(f"{storage_path}/{file_name.replace(' ', '_')}.pkl", "wb") as f:
        pickle.dump(vector_store, f)
    
    # Clean up temp file
    os.unlink(pdf_path)
    
    return vector_store, chapters

# Generate chapter summaries with Groq
def generate_summaries(chapters, learning_style, groq_client, model):
    summaries = {}
    
    for i, chapter in enumerate(chapters):
        # Combine all text in the chapter
        chapter_text = " ".join([doc.page_content for doc in chapter])
        
        # Generate prompt based on learning style
        prompt = f"""
        Summarize the following text for a {learning_style}:
        
        {chapter_text[:3000]}  # Limiting to first 3000 chars for API efficiency
        
        Give a summary that includes:
        1. Main concepts in bullet points
        2. A visual metaphor or analogy
        3. Key takeaways
        """
        
        # Call Groq API
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            summary = chat_completion.choices[0].message.content
            summaries[f"Chapter {i+1}"] = summary
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            summaries[f"Chapter {i+1}"] = "Error generating summary."
    
    return summaries

# File upload section
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Process the uploaded file
if uploaded_file and GROQ_API_KEY:
    # Initialize Groq client
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    file_name = uploaded_file.name.split('.')[0]
    
    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            # Process the PDF and get vector store and chapters
            vector_store, chapters = process_pdf(uploaded_file, file_name)
            
            # Save to session state
            st.session_state.current_vector_store = vector_store
            st.session_state.current_file_name = file_name
            
            # Generate summaries
            with st.spinner("Generating chapter summaries..."):
                summaries = generate_summaries(
                    chapters,
                    learning_style,
                    groq_client,
                    llm_model
                )
                st.session_state.chapter_summaries = summaries
            
            # Add to processed files if not already there
            if file_name not in st.session_state.processed_files:
                st.session_state.processed_files.append(file_name)
            
            st.success(f"Successfully processed {file_name}!")

# Display processed files
if st.session_state.processed_files:
    st.header("Your Library")
    
    selected_file = st.selectbox(
        "Select a document to explore:",
        st.session_state.processed_files
    )
    
    # Load vector store if needed
    if selected_file != st.session_state.current_file_name:
        storage_path = f"temp_storage/{st.session_state.user_id}"
        vector_store_path = f"{storage_path}/{selected_file.replace(' ', '_')}.pkl"
        
        if os.path.exists(vector_store_path):
            with open(vector_store_path, "rb") as f:
                st.session_state.current_vector_store = pickle.load(f)
            st.session_state.current_file_name = selected_file
        else:
            st.error("Vector store not found. Please reprocess the document.")

# Display chapter summaries
if st.session_state.chapter_summaries:
    st.header("Chapter Summaries")
    
    for chapter, summary in st.session_state.chapter_summaries.items():
        with st.expander(chapter):
            st.markdown(summary)

# Q&A section
if st.session_state.current_vector_store and GROQ_API_KEY:
    st.header("Ask Questions About Your Document")
    
    question = st.text_input("Ask a question about the content:")
    
    if question and st.button("Get Answer"):
        with st.spinner("Generating answer..."):
            # Initialize Groq client
            groq_client = Groq(api_key=GROQ_API_KEY)
            
            # Search for relevant documents
            docs = st.session_state.current_vector_store.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate prompt
            prompt = f"""
            Answer the following question based on the provided context.
            
            Context:
            {context}
            
            Question: {question}
            
            For a {learning_style}, provide:
            1. A clear, concise answer
            2. An example or illustration if applicable
            3. A connection to any main concepts from the document
            """
            
            # Call Groq API
            try:
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=llm_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                answer = chat_completion.choices[0].message.content
                
                st.markdown("### Answer")
                st.markdown(answer)
                
                # Show sources
                with st.expander("Sources"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Source {i+1}**")
                        st.markdown(doc.page_content)
                        st.markdown(f"*Page: {doc.metadata.get('page', 'Unknown')}*")
                        st.divider()
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

# App footer
st.markdown("---")

st.markdown("BookScribe AI - Powered by Groq and Hugging Face")


