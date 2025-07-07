# GenAI RAG Assistant

A modular Retrieval-Augmented Generation (RAG) assistant that leverages LangChain, Hugging Face Transformers, and FAISS to deliver context-aware answers from PDF documents and web sources. Built for scalable enterprise use with support for local and cloud-based LLMs.

## Features

- Document ingestion with automatic chunking and embedding
- Vector search using FAISS
- Integration with OpenAI, Mistral, and other Hugging Face models
- Query handling via LangChain-based retrieval pipeline
- Streamlit or FastAPI interface for interactive usage
- 
## Tech Stack

- Python
- LangChain
- Hugging Face Transformers
- FAISS
- OpenAI / Mistral / Ollama
- Streamlit or FastAPI

## Project Structure

```
genai-rag-assistant/
│
├── app/              # Streamlit or FastAPI code
├── data/             # Input documents (PDF, TXT)
├── embeddings/       # FAISS index storage
├── models/           # LLM configs or wrappers
├── utils/            # Helper functions
├── README.md
└── requirements.txt  # Project dependencies
```



## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Afiq77/genai-rag-assistant.git
   cd genai-rag-assistant
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app/main.py
   # or for FastAPI:
   uvicorn app.main:app --reload

## Future Improvements

- Add voice input using Whisper
- Add user query analytics and feedback loop
- PostgreSQL integration for session tracking
- Enhanced document preview and source highlighting


## License

This project is licensed under the MIT License.


