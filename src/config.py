"""
Configuration file for NIU Admission Knowledge Base
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Pinecone Settings
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "niu-admission-knowledge-base")
    
    # Ollama Settings
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text:latest")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    # Text Processing Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))
    INDEX_DIMENSION=768
    
    # File Paths
    DATA_DIR = "data"
    PDF_FILE = "Noida International University.pdf"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        required_vars = [
            ("PINECONE_API_KEY", cls.PINECONE_API_KEY),
        ]
        
        missing_vars = []
        for var_name, var_value in required_vars:
            if not var_value:
                missing_vars.append(var_name)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True