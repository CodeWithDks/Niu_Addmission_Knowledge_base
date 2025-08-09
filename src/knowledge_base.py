"""
NIU Admission Knowledge Base Builder
This module handles the complete pipeline for creating and managing the knowledge base
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import uuid
from typing import List, Dict, Any

# Document processing imports
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Vector store and embedding imports
import pinecone
from pinecone import Pinecone, ServerlessSpec
import ollama
import numpy as np
from contextlib import suppress

# Local imports
from .config import Config
from .utils import (
    setup_logging, create_directories, generate_doc_id, 
    validate_chunks, display_progress, print_success, 
    print_error, print_warning, print_info
)

class NIUKnowledgeBase:
    """
    Main class for building and managing NIU Admission Knowledge Base
    """
    
    def __init__(self):
        """Initialize the knowledge base builder"""
        self.logger = setup_logging(Config.LOG_LEVEL)
        self.config = Config()
        
        # Validate configuration
        try:
            self.config.validate_config()
            print_success("Configuration validated successfully")
        except ValueError as e:
            print_error(f"Configuration error: {e}")
            raise
        
        # Initialize components
        self.text_splitter = None
        self.pinecone_client = None
        self.index = None
        self.embedding_model = Config.OLLAMA_MODEL
        
        print_info("NIU Knowledge Base initialized")

    def setup_text_splitter(self):
        """Initialize the text splitter for chunking documents"""
        print_info("Setting up text splitter...")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                "? ",    # Question endings
                "! ",    # Exclamation endings
                "; ",    # Semicolon
                ", ",    # Comma
                " ",     # Spaces
                ""       # Characters
            ]
        )
        print_success("Text splitter configured")

    def load_pdf_document(self, file_path: str) -> List[Document]:
        """
        Load and extract text from PDF document
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            List[Document]: List of document objects
        """
        print_info(f"Loading PDF document: {file_path}")
        
        if not os.path.exists(file_path):
            print_error(f"PDF file not found: {file_path}")
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        documents = []
        
        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(file_path)
            
            with display_progress("Extracting text from PDF...") as progress:
                task = progress.add_task("Processing pages...", total=len(pdf_document))
                
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    text = page.get_text()
                    
                    if text.strip():  # Only add non-empty pages
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": file_path,
                                "page": page_num + 1,
                                "total_pages": len(pdf_document),
                                "document_type": "NIU_Admission_Guide"
                            }
                        )
                        documents.append(doc)
                    
                    progress.update(task, advance=1)
            
            pdf_document.close()
            print_success(f"Successfully loaded {len(documents)} pages from PDF")
            
        except Exception as e:
            print_error(f"Error loading PDF: {str(e)}")
            raise
        
        return documents

    def split_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Split documents into smaller chunks for better retrieval
        
        Args:
            documents (List[Document]): List of documents to split
            
        Returns:
            List[Dict[str, Any]]: List of text chunks with metadata
        """
        print_info("Splitting documents into chunks...")
        
        if not self.text_splitter:
            self.setup_text_splitter()
        
        chunks = []
        
        with display_progress("Splitting documents...") as progress:
            task = progress.add_task("Processing documents...", total=len(documents))
            
            for doc in documents:
                # Split the document
                splits = self.text_splitter.split_documents([doc])
                
                for i, split in enumerate(splits):
                    chunk = {
                        "text": split.page_content,
                        "metadata": {
                            **split.metadata,
                            "chunk_id": i,
                            "total_chunks": len(splits),
                            "doc_id": generate_doc_id(split.page_content, split.metadata)
                        }
                    }
                    chunks.append(chunk)
                
                progress.update(task, advance=1)
        
        # Validate and clean chunks
        valid_chunks = validate_chunks(chunks)
        
        print_success(f"Created {len(valid_chunks)} valid chunks from {len(documents)} documents")
        return valid_chunks

    def setup_pinecone(self):
        """Initialize Pinecone vector database"""
        print_info("Setting up Pinecone connection...")
        
        try:
            # Initialize Pinecone
            self.pinecone_client = Pinecone(api_key=self.config.PINECONE_API_KEY)
            
            # Check if index exists
            existing_indexes = self.pinecone_client.list_indexes()
            index_names = [index.name for index in existing_indexes.indexes]
            
            if self.config.PINECONE_INDEX_NAME not in index_names:
                print_info(f"Creating new Pinecone index: {self.config.PINECONE_INDEX_NAME}")
                
                # Create index with appropriate dimensions for nomic-embed-text
                self.pinecone_client.create_index(
                    name=self.config.PINECONE_INDEX_NAME,
                    dimension=768,  # nomic-embed-text embedding dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                
                # Wait for index to be ready
                print_info("Waiting for index to be ready...")
                time.sleep(30)
            
            # Connect to the index
            self.index = self.pinecone_client.Index(self.config.PINECONE_INDEX_NAME)
            print_success("Pinecone setup completed")
            
        except Exception as e:
            print_error(f"Error setting up Pinecone: {str(e)}")
            raise

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Ollama's nomic-embed-text model.
        Relies on outer function to handle any progress display.
        """
        embeddings = []

        if not texts:
            self.logger.warning("No texts provided for embedding")
            return embeddings

        for text in texts:
            try:
                response = ollama.embeddings(
                    model=self.embedding_model,
                    prompt=text
                )

                # Handle both possible Ollama response formats
                if isinstance(response, dict):
                    if "embedding" in response:
                        vector = response["embedding"]
                    elif "data" in response and len(response["data"]) > 0:
                        vector = response["data"][0].get("embedding", [])
                    else:
                        vector = []
                else:
                    vector = []

                # Validate vector length
                if not vector or len(vector) != 768:
                    self.logger.error(
                        f"Invalid embedding length ({len(vector) if vector else 0}) for text: '{text[:50]}...'"
                    )
                    vector = [0.0] * 768

                embeddings.append(vector)

            except Exception as e:
                self.logger.error(f"Error generating embedding: {str(e)}")
                embeddings.append([0.0] * 768)

            # Small delay to avoid overwhelming Ollama
            time.sleep(0.1)

        print_success(f"Generated {len(embeddings)} embeddings")
        return embeddings


    def store_in_pinecone(self, chunks: List[Dict[str, Any]]):
        """
        Store chunks and their embeddings in Pinecone

        Args:
            chunks (List[Dict[str, Any]]): List of text chunks to store
        """
        print_info("Storing embeddings in Pinecone...")

        if not self.index:
            self.setup_pinecone()

        batch_size = self.config.BATCH_SIZE
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        with display_progress("Uploading to Pinecone...") as progress:
            task = progress.add_task("Uploading batches...", total=total_batches)

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                # Extract texts for embedding
                texts = [chunk['text'] for chunk in batch]

                # Generate embeddings
                embeddings = self.get_embeddings(texts)

                # Prepare vectors for Pinecone
                vectors = []
                for chunk, embedding in zip(batch, embeddings):
                    vector_id = f"{chunk['metadata']['doc_id']}_{chunk['metadata'].get('chunk_id', 0)}"
                    vector = {
                        'id': vector_id,
                        'values': embedding,
                        'metadata': {
                            'text': chunk['text'][:1000],
                            'source': chunk['metadata'].get('source', ''),
                            'page': chunk['metadata'].get('page', 0),
                            'chunk_id': chunk['metadata'].get('chunk_id', 0),
                            'document_type': chunk['metadata'].get('document_type', '')
                        }
                    }
                    vectors.append(vector)

                # ✅ Remove duplicate IDs before uploading
                unique_vectors = {v['id']: v for v in vectors}
                vectors = list(unique_vectors.values())

                # Upload to Pinecone
                try:
                    self.index.upsert(vectors=vectors)
                    progress.update(task, advance=1)
                except Exception as e:
                    print_error(f"Error uploading batch {i // batch_size + 1}: {str(e)}")
                    continue

                time.sleep(1)  # avoid hitting rate limits

        print_success(f"Successfully stored {len(chunks)} unique chunks in Pinecone")


    def build_knowledge_base(self, pdf_path: Optional[str] = None):
        """
        Complete pipeline to build the knowledge base
        
        Args:
            pdf_path (Optional[str]): Path to PDF file, uses default if None
        """
        print_info("Starting NIU Knowledge Base build process...")
        
        try:
            # Step 1: Setup directories
            create_directories()
            
            # Step 2: Determine PDF path
            if pdf_path is None:
                pdf_path = os.path.join(self.config.DATA_DIR, self.config.PDF_FILE)
            
            # Step 3: Load PDF document
            documents = self.load_pdf_document(pdf_path)
            
            # Step 4: Split documents into chunks
            chunks = self.split_documents(documents)
            
            # Step 5: Setup Pinecone
            self.setup_pinecone()
            
            # Step 6: Store in Pinecone
            self.store_in_pinecone(chunks)
            
            # Step 7: Verify the build
            self.verify_knowledge_base()
            
            print_success("🎉 Knowledge Base build completed successfully!")
            
        except Exception as e:
            print_error(f"Error building knowledge base: {str(e)}")
            self.logger.error(f"Build failed: {str(e)}", exc_info=True)
            raise

    def verify_knowledge_base(self):
        """Verify that the knowledge base was built correctly"""
        print_info("Verifying knowledge base...")
        
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            
            print_success(f"Index contains {stats['total_vector_count']} vectors")
            print_info(f"Index dimension: {stats['dimension']}")
            
            # Test query
            test_query = "What programs are offered at NIU?"
            test_embedding = self.get_embeddings([test_query])[0]
            
            results = self.index.query(
                vector=test_embedding,
                top_k=3,
                include_metadata=True
            )
            
            if results['matches']:
                print_success("Knowledge base is working correctly!")
                print_info(f"Test query returned {len(results['matches'])} results")
            else:
                print_warning("Knowledge base created but test query returned no results")
                
        except Exception as e:
            print_error(f"Error verifying knowledge base: {str(e)}")

    def query_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the knowledge base for relevant information
        
        Args:
            query (str): Query text
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of relevant documents
        """
        if not self.index:
            self.setup_pinecone()
        
        # Generate embedding for query
        query_embedding = self.get_embeddings([query])[0]
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results['matches']