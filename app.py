"""
RAG Chatbot - Streamlit Application
Complete, ready-to-use app.py file
Just copy this entire code into a file named app.py
"""

import streamlit as st
from pathlib import Path
import logging

import config
import db
import ingest
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None


@st.cache_resource
def get_embeddings_cached():
    """Cache embeddings model"""
    return ingest.get_embeddings()


def get_llm():
    """Initialize Groq LLM"""
    return ChatGroq(
        groq_api_key=config.GROQ_API_KEY,
        model_name=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS
    )


def main():
    st.set_page_config(
        page_title="RAG Chatbot POC",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title(" RAG Chatbot - using Langchain")
    st.markdown("Upload documents and chat with your data using AI")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # User login
        username_input = st.text_input(
            "Username",
            key="username_input",
            placeholder="Enter your username"
        )
        
        if username_input:
            if not st.session_state.user_id or st.session_state.username != username_input:
                try:
                    st.session_state.user_id = db.get_or_create_user(username_input)
                    st.session_state.username = username_input
                    st.success(f"✅ Logged in as **{username_input}**")
                    logger.info(f"User logged in: {username_input} (ID: {st.session_state.user_id})")
                except Exception as e:
                    st.error(f"Login failed: {e}")
        
        st.divider()
        
        # Document upload section
        if st.session_state.user_id:
            st.subheader("📄 Document Upload")
            
            uploaded_file = st.file_uploader(
                "Upload PDF Document",
                type=['pdf'],
                help="Upload a PDF to index and query"
            )
            
            if uploaded_file:
                st.info(f"Selected: **{uploaded_file.name}**")
                
                if st.button("📥 Process Document", type="primary"):
                    try:
                        # Save uploaded file
                        user_upload_dir = config.get_user_upload_dir(str(st.session_state.user_id))
                        save_path = user_upload_dir / uploaded_file.name
                        
                        with open(save_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        
                        logger.info(f"Saved file: {save_path}")
                        
                        # Process document
                        with st.spinner("🔄 Processing document... This may take a minute."):
                            # Get cached embeddings
                            if not st.session_state.embeddings:
                                st.session_state.embeddings = get_embeddings_cached()
                            
                            # Ingest PDF
                            vectordb = ingest.ingest_pdf(
                                str(save_path),
                                str(st.session_state.user_id)
                            )
                            
                            st.session_state.vectordb = vectordb
                            
                            count = vectordb._collection.count()
                            st.success(f"✅ Successfully indexed **{uploaded_file.name}**!")
                            st.info(f"📊 Created {count} document chunks")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing document: {e}")
                        logger.error(f"Document processing error: {e}", exc_info=True)
        else:
            st.warning("👆 Please enter a username to continue")
        
        st.divider()
        
        # Database status
        st.subheader("📊 Status")
        
        if st.session_state.vectordb:
            try:
                count = st.session_state.vectordb._collection.count()
                st.success(f"✅ Database: **{count}** chunks")
            except:
                st.warning("⚠️ Database loaded but status unavailable")
        else:
            st.info("ℹ️ No documents indexed yet")
        
        # Clear chat button
        if st.session_state.user_id:
            if st.button("🗑️ Clear Chat History", type="secondary"):
                try:
                    deleted = db.clear_chat_history(st.session_state.user_id)
                    st.success(f"Cleared {deleted} messages")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing history: {e}")
    
    # Main chat area
    if not st.session_state.user_id:
        st.info("👈 Please enter a username in the sidebar to start chatting")
        st.markdown("""
        ### Welcome to RAG Chatbot!
        
        **Getting Started:**
        1. Enter a username in the sidebar
        2. Upload a PDF document
        3. Click "Process Document"
        4. Start asking questions about your document
        
        **Features:**
        - 📄 PDF document processing
        - 🔍 Semantic search across your documents
        - 💬 Natural conversation with AI
        - 📚 Source citations for answers
        """)
        return
    
    # Load vectordb if exists but not loaded
    if not st.session_state.vectordb:
        try:
            if not st.session_state.embeddings:
                st.session_state.embeddings = get_embeddings_cached()
            
            st.session_state.vectordb = ingest.load_vectordb(
                str(st.session_state.user_id),
                st.session_state.embeddings
            )
        except Exception as e:
            logger.error(f"Error loading vectordb: {e}")
    
    # Display chat history
    try:
        messages = db.get_chat_history(st.session_state.user_id)
        
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
        logger.error(f"Chat history error: {e}")
    
    # Chat input
    if query := st.chat_input("Ask anything about your documents..."):
        
        # Check if vectordb exists
        if not st.session_state.vectordb:
            st.error("⚠️ Please upload and process a document first!")
            return
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Save user message
        try:
            db.save_message(st.session_state.user_id, "user", query)
        except Exception as e:
            logger.error(f"Error saving user message: {e}")
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Initialize LLM
                    llm = get_llm()
                    
                    # Create retriever
                    retriever = ingest.create_retriever(st.session_state.vectordb)
                    
                    # Create prompt template
                    prompt = PromptTemplate(
                        template=config.SYSTEM_PROMPT,
                        input_variables=["context", "question"]
                    )
                    
                    # Create QA chain
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": prompt}
                    )
                    
                    # Get response
                    result = qa_chain.invoke({"query": query})
                    answer = result["result"]
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Save assistant message
                    try:
                        db.save_message(st.session_state.user_id, "assistant", answer)
                    except Exception as e:
                        logger.error(f"Error saving assistant message: {e}")
                    
                    # Show sources
                    if result.get("source_documents"):
                        with st.expander(" View Source Documents"):
                            for i, doc in enumerate(result["source_documents"], 1):
                                st.markdown(f"**Source {i}:**")
                                content = doc.page_content
                                if len(content) > 300:
                                    content = content[:300] + "..."
                                st.text(content)
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    st.caption(f"Metadata: {doc.metadata}")
                                st.divider()
                    
                except Exception as e:
                    error_msg = f" Error generating response: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Response generation error: {e}", exc_info=True)
                    
                    # Save error message
                    try:
                        db.save_message(st.session_state.user_id, "assistant", error_msg)
                    except:
                        pass


if __name__ == "__main__":
    main()