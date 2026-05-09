# RAG PDF Chatbot using LangChain, Groq, ChromaDB & Streamlit

## Project Overview

This project is a Retrieval-Augmented Generation (RAG) based PDF chatbot that allows users to upload PDF documents and ask questions related to the uploaded content.

The chatbot uses:
- LangChain for orchestration
- Groq LLM for response generation
- HuggingFace embeddings for semantic search
- ChromaDB as vector database
- Streamlit for frontend UI
- Docker for containerization

The system retrieves relevant information from uploaded PDFs and generates context-aware answers.

---


##  Live Demo  
Try out the deployed project here:  

-  **Streamlit App** → [https://agnetic-ai-chatbot.streamlit.app/](https://ragchatbot-bbcvfwoztjycthwd6uw4xb.streamlit.app/)  

---

# Features

- Upload PDF documents
- Extract PDF text
- Chunk large documents
- Generate embeddings
- Store embeddings in ChromaDB
- Semantic similarity retrieval
- Context-aware AI responses
- Hallucination control
- Streamlit web interface
- Dockerized deployment
- Public deployment support

---

# Tech Stack

| Technology | Usage |
|------------|-------|
| Python | Programming Language |
| Streamlit | Frontend UI |
| LangChain | RAG Pipeline |
| Groq | LLM Provider |
| HuggingFace | Embeddings |
| ChromaDB | Vector Database |
| PyPDF | PDF Processing |
| Docker | Containerization |

---

# RAG Architecture

```text
PDF Upload
    ↓
PyPDFLoader
    ↓
Text Chunking
    ↓
Embeddings Generation
    ↓
Chroma Vector Database
    ↓
Retriever
    ↓
Prompt Template
    ↓
Groq LLM
    ↓
Final Response
```

---

# Project Structure

```text
RAG_Project/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
└── sample.pdf
```

---

# Installation & Local Setup

## Step 1: Clone Repository

```bash
git clone <your_repository_url>
cd rag-chatbot
```

---

## Step 2: Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4: Create .env File

Create a `.env` file and add:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## Step 5: Run Streamlit App

```bash
streamlit run app.py
```

---

# Docker Setup

## Build Docker Image

```bash
docker build -t rag-chatbot .
```

---

## Run Docker Container

```bash
docker run -p 8501:8501 rag-chatbot
```

---

# Streamlit Cloud Deployment

1. Push project to GitHub
2. Open Streamlit Community Cloud
3. Create new app
4. Select GitHub repository
5. Add GROQ_API_KEY inside Streamlit Secrets
6. Deploy application

---

# Example Questions

After uploading PDF, ask questions like:

- What projects are mentioned in the resume?
- What skills are available in the document?
- Summarize the uploaded PDF.
- What technologies are used in this project?

---

# Hallucination Control

If information is not available in uploaded PDF, the chatbot responds:

```text
Answer not found in the uploaded PDF.
```

This helps reduce hallucinations and ensures grounded responses.

---

# Future Improvements

- Multi-PDF support
- Chat history
- Memory integration
- FAISS vector database
- Source citations
- Authentication
- Cloud database integration
- LangSmith tracing
- LangGraph agents

---

# Author

Yadagiri

---

# License

This project is for educational and learning purposes.
