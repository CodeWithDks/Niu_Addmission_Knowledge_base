#!/usr/bin/env python3
"""
NIU Admission Knowledge Base Setup Script
Run this script to build the complete knowledge base
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.knowledge_base import NIUKnowledgeBase
from src.utils import print_success, print_error, print_info
from src.config import Config

def check_ollama_availability():
    """Check if Ollama is running and model is available"""
    import ollama
    
    try:
        # Check if Ollama is running
        models = ollama.list()
        available_models = [model['name'] for model in models['models']]
        
        if Config.OLLAMA_MODEL not in available_models:
            print_error(f"Model {Config.OLLAMA_MODEL} not found in Ollama")
            print_info("Available models:")
            for model in available_models:
                print(f"  - {model}")
            print_info(f"Please run: ollama pull {Config.OLLAMA_MODEL}")
            return False
        
        print_success("Ollama is running and model is available")
        return True
        
    except Exception as e:
        print_error(f"Ollama connection error: {str(e)}")
        print_info("Please make sure Ollama is running: ollama serve")
        return False

def check_pdf_file():
    """Check if the PDF file exists"""
    pdf_path = os.path.join(Config.DATA_DIR, Config.PDF_FILE)
    
    if not os.path.exists(pdf_path):
        print_error(f"PDF file not found: {pdf_path}")
        print_info("Please ensure the NIU admission PDF is in the data/ directory")
        return False
    
    print_success(f"PDF file found: {pdf_path}")
    return True

def main():
    """Main function to setup the knowledge base"""
    parser = argparse.ArgumentParser(description='Build NIU Admission Knowledge Base')
    parser.add_argument('--pdf-path', type=str, help='Custom path to PDF file')
    parser.add_argument('--force-rebuild', action='store_true', help='Force rebuild even if index exists')
    parser.add_argument('--test-only', action='store_true', help='Only run tests without building')
    args = parser.parse_args()
    
    print_info("🚀 NIU Admission Knowledge Base Setup")
    print_info("=" * 50)
    
    # Pre-flight checks
    print_info("Running pre-flight checks...")
    
    # Check environment file
    if not os.path.exists('.env'):
        print_error(".env file not found")
        print_info("Please create .env file with required configuration")
        return 1
    
    # Check Ollama
    if not check_ollama_availability():
        return 1
    
    # Check PDF file (unless custom path provided)
    if not args.pdf_path and not check_pdf_file():
        return 1
    
    print_success("✅ All pre-flight checks passed")
    
    if args.test_only:
        print_info("Running in test-only mode")
        try:
            kb = NIUKnowledgeBase()
            kb.setup_pinecone()
            kb.verify_knowledge_base()
            return 0
        except Exception as e:
            print_error(f"Test failed: {str(e)}")
            return 1
    
    # Build knowledge base
    try:
        print_info("Initializing knowledge base builder...")
        kb = NIUKnowledgeBase()
        
        print_info("Starting knowledge base build process...")
        kb.build_knowledge_base(pdf_path=args.pdf_path)
        
        print_success("🎉 Knowledge base setup completed successfully!")
        print_info("\nNext steps:")
        print_info("1. Your knowledge base is now ready for use")
        print_info("2. You can query it using the NIUKnowledgeBase.query_knowledge_base() method")
        print_info("3. Consider building a chatbot interface on top of this knowledge base")
        
        return 0
        
    except KeyboardInterrupt:
        print_warning("Setup interrupted by user")
        return 1
    except Exception as e:
        print_error(f"Setup failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)