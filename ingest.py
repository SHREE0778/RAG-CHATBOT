"""
Document ingestion and vectorization module
"""

from typing import List, Optional
from pathlib import Path
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import config

logger = logging.getLogger(__name__)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize and return HuggingFace embeddings model"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': config.EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': config.NORMALIZE_EMBEDDINGS}
        )
        logger.info(f"Loaded embedding model: {config.EMBEDDING_MODEL}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        raise RuntimeError(f"Embedding model initialization failed: {e}")


def load_pdf(pdf_path: str) -> List[Document]:
    """Load PDF document"""
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    try:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        
        if not documents:
            raise ValueError("PDF loaded but contains no content")
        
        logger.info(f"Loaded PDF: {pdf_path.name} ({len(documents)} pages)")
        return documents
    except Exception as e:
        logger.error(f"Failed to load PDF {pdf_path}: {e}")
        raise ValueError(f"Could not load PDF: {e}")


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=config.SEPARATORS
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")
    
    return chunks


def create_vectordb(
    documents: List[Document],
    user_id: str,
    embeddings: Optional[HuggingFaceEmbeddings] = None
) -> Chroma:
    """Create vector database from documents"""
    if not documents:
        raise ValueError("Cannot create vectorDB from empty document list")
    
    if embeddings is None:
        embeddings = get_embeddings()
    
    persist_dir = config.get_user_persist_dir(user_id)
    
    try:
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(persist_dir),
            collection_name=config.COLLECTION_NAME
        )
        
        count = vectordb._collection.count()
        logger.info(f"Created vectorDB for user {user_id}: {count} documents")
        
        return vectordb
    except Exception as e:
        logger.error(f"Failed to create vectorDB: {e}")
        raise RuntimeError(f"VectorDB creation failed: {e}")


def load_vectordb(
    user_id: str,
    embeddings: Optional[HuggingFaceEmbeddings] = None
) -> Optional[Chroma]:
    """Load existing vector database for user"""
    persist_dir = config.get_user_persist_dir(user_id)
    
    if not persist_dir.exists():
        logger.info(f"No vectorDB found for user {user_id}")
        return None
    
    if not any(persist_dir.iterdir()):
        logger.info(f"VectorDB directory empty for user {user_id}")
        return None
    
    if embeddings is None:
        embeddings = get_embeddings()
    
    try:
        vectordb = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name=config.COLLECTION_NAME
        )
        
        count = vectordb._collection.count()
        if count == 0:
            logger.warning(f"VectorDB exists but empty for user {user_id}")
            return None
        
        logger.info(f"Loaded vectorDB for user {user_id}: {count} documents")
        return vectordb
    except Exception as e:
        logger.error(f"Failed to load vectorDB for user {user_id}: {e}")
        return None


def ingest_pdf(pdf_path: str, user_id: str) -> Chroma:
    """Complete ingestion pipeline: load PDF, chunk, create vectorDB"""
    logger.info(f"Starting ingestion for user {user_id}: {pdf_path}")
    
    documents = load_pdf(pdf_path)
    chunks = chunk_documents(documents)
    vectordb = create_vectordb(chunks, user_id)
    
    logger.info(f"Ingestion complete for user {user_id}")
    return vectordb


def create_retriever(vectordb: Chroma, k: Optional[int] = None):
    """Create retriever from vector database"""
    if k is None:
        k = config.RETRIEVAL_K
    
    retriever = vectordb.as_retriever(
        search_type=config.SEARCH_TYPE,
        search_kwargs={"k": k}
    )
    
    logger.info(f"Created retriever with k={k}")
    return retriever