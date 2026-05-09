import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import os

# Load environment variables
load_dotenv()

# Get Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Create Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

# Streamlit UI
st.title("RAG PDF Chat Application")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

# Check PDF upload
if uploaded_file is not None:

    st.success("PDF Uploaded Successfully")

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())

        temp_pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(temp_pdf_path)

    documents = loader.load()

    # Text Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    st.write(f"Total Chunks Created: {len(chunks)}")

    # Create Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create Vector Database
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Create Retriever
    retriever = vectorstore.as_retriever()

    # User Question
    question = st.text_input(
        "Ask a question from the PDF"
    )

    # Generate Response Button
    if st.button("Generate Response"):

        if question:

            # Retrieve relevant chunks
            retrieved_docs = retriever.invoke(question)

            # Combine retrieved context
            context = "\n\n".join(
                [doc.page_content for doc in retrieved_docs]
            )

            # Prompt Template
            prompt = ChatPromptTemplate.from_template(
                """
                You are a helpful AI assistant.

                Answer the question ONLY from the provided context.

                If the answer is not available in the context,
                say:
                "Answer not found in the uploaded PDF."

                Context:
                {context}

                Question:
                {question}
                """
            )

            # Final Prompt
            final_prompt = prompt.format(
                context=context,
                question=question
            )

            # Generate Response
            response = llm.invoke(final_prompt)

            # Display Response
            st.subheader("RAG Response")

            st.write(response.content)