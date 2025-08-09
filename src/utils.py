"""
Utility functions for NIU Admission Knowledge Base
"""
import logging
import os
import hashlib
from typing import List, Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('knowledge_base.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'logs', 'output']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            console.print(f"✅ Created directory: {directory}")

def generate_doc_id(text: str, metadata: Dict[str, Any]) -> str:
    """Generate unique document ID based on content and metadata"""
    content = f"{text}{str(metadata)}"
    return hashlib.md5(content.encode()).hexdigest()

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove special characters that might cause issues
    text = text.replace('\x00', '')  # Remove null characters
    text = text.replace('\ufffd', '')  # Remove replacement characters
    
    return text.strip()

def validate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and filter chunks before embedding"""
    valid_chunks = []
    
    for chunk in chunks:
        # Skip empty or very short chunks
        if len(chunk.get('text', '').strip()) < 50:
            continue
            
        # Clean the text
        chunk['text'] = clean_text(chunk['text'])
        
        # Ensure metadata exists
        if 'metadata' not in chunk:
            chunk['metadata'] = {}
            
        valid_chunks.append(chunk)
    
    return valid_chunks

def display_progress(description: str):
    """Create a progress display context manager"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    )

def print_success(message: str):
    """Print success message with styling"""
    console.print(f"✅ {message}", style="green")

def print_error(message: str):
    """Print error message with styling"""
    console.print(f"❌ {message}", style="red")

def print_warning(message: str):
    """Print warning message with styling"""
    console.print(f"⚠️  {message}", style="yellow")

def print_info(message: str):
    """Print info message with styling"""
    console.print(f"ℹ️  {message}", style="blue")