# 🎓 NIU Admission Knowledge Base

A comprehensive AI-powered knowledge base for **Noida International University (NIU) admissions** — built using **Ollama embeddings** and **Pinecone vector storage** for lightning-fast and accurate information retrieval.

---

## ✨ Features

- 📄 **PDF Document Processing** – Extracts and processes NIU admission guide PDF.  
- ✂️ **Smart Text Chunking** – Splits documents into optimal chunks for better semantic search.  
- 🤖 **Ollama Embeddings** – Uses `nomic-embed-text:latest` model for high-quality vector embeddings.  
- 📦 **Pinecone Vector Storage** – Scalable, cloud-based vector DB for similarity search.  
- 🛠 **Comprehensive Logging** – Debug and monitor every step with structured logs.  
- 🎨 **Rich CLI Interface** – Beautiful progress bars and status updates with `rich`.  
- ⚡ **Fast Retrieval** – Get relevant admission details in milliseconds.  
- 🌐 **Easy Integration** – Works with Streamlit, FastAPI, Flask, or chatbot workflows.  

---

## 📂 Project Structure

```

niu-admission-knowledge-base/
│
├── data/
│   └── Noida International University.pdf      # Admission guide PDF
│
├── src/
│   ├── **init**.py
│   ├── knowledge\_base.py                        # Core KB builder & query system
│   ├── config.py                                # Configurations & constants
│   └── utils.py                                 # Helper functions & utilities
│
├── .env                                         # Environment variables
├── .gitignore                                   # Ignore unnecessary files
├── requirements.txt                             # Python dependencies
├── setup\_knowledge\_base.py                      # CLI setup script for KB
└── README.md

````

---

## 📋 Prerequisites

1. **Python 3.8+** installed.  
2. **Ollama** installed and running:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh

   # Pull the embedding model
   ollama pull nomic-embed-text:latest

   # Start Ollama server
   ollama serve
````

3. **Pinecone Account** with API key:

   * Sign up at [https://www.pinecone.io](https://www.pinecone.io)
   * Create an index (e.g., `niu-admission-kb`) with:

     * **Dimension:** 768 (matching embedding model output)
     * **Metric:** cosine
     * **Pod Type:** p1/x1 or s1 (for testing)

---

## ⚙️ Installation

1. **Clone this repository**:

   ```bash
   git clone https://github.com/your-username/niu-admission-knowledge-base.git
   cd niu-admission-knowledge-base
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** in `.env`:

   ```env
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=your_pinecone_environment_here
   PINECONE_INDEX=niu-admission-kb
   OLLAMA_BASE_URL=http://localhost:11434
   EMBEDDING_MODEL=nomic-embed-text:latest
   ```

---

## 🚀 Usage

### 1. **Build the Knowledge Base**

```bash
python setup_knowledge_base.py
```

This will:

* Load the admission PDF
* Split it into chunks
* Generate embeddings with Ollama
* Store them in Pinecone

---

### 2. **Query the Knowledge Base (Python)**

```python
from src.knowledge_base import NIUKnowledgeBase

kb = NIUKnowledgeBase()
response = kb.query_knowledge_base("What are the requirements for MBA admission?")
print(response)
```

---

### 3. **Test with Streamlit**

Create `app.py`:

```python
import streamlit as st
from src.knowledge_base import NIUKnowledgeBase

st.title("🎓 NIU Admission Knowledge Base")
query = st.text_input("Ask a question about NIU admissions:")
if st.button("Search"):
    kb = NIUKnowledgeBase()
    results = kb.query_knowledge_base(query)
    for idx, result in enumerate(results, 1):
        st.markdown(f"**Result {idx}:** {result['text']}")
```

Run:

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. **Document Loading** → Reads the NIU admission guide PDF.
2. **Text Chunking** → Splits text into overlapping segments for better retrieval.
3. **Embedding Generation** → Converts each chunk into a 768-dimensional vector using Ollama.
4. **Vector Storage** → Saves vectors into Pinecone for fast similarity search.
5. **Query Processing** → Converts user query into an embedding and retrieves top matches from Pinecone.

---

## 📌 Example Queries

* `"What is the fee structure for B.Tech?"`
* `"When is the admission deadline?"`
* `"Tell me about NIU MBA program."`

---

## 🛡 License

This project is licensed under the **MIT License** – feel free to modify and use it.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## 📧 Contact

For inquiries, contact: **[admissions@niu.edu.in](mailto:admissions@niu.edu.in)** or visit [NIU Official Website](https://niu.edu.in)
