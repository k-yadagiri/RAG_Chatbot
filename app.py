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

# ========================= PAGE CONFIG =========================

st.set_page_config(
    page_title="AI RAG PDF Assistant",
    page_icon="🤖",
    layout="wide"
)

# ========================= CUSTOM CSS =========================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #111827
    );
    color: white;
}

/* Hide Streamlit Header */
header {
    visibility: hidden;
}

/* Main Title */
.main-title {
    font-size: 60px;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(
        90deg,
        #38bdf8,
        #818cf8,
        #c084fc
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 20px;
    margin-bottom: 10px;
}

/* Subtitle */
.sub-text {
    text-align: center;
    color: #cbd5e1;
    font-size: 20px;
    margin-bottom: 40px;
}

/* Upload Box */
section[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
}

/* Text Input */
.stTextInput > div > div > input {
    background-color: rgba(255,255,255,0.08);
    color: white;
    border-radius: 14px;
    border: 1px solid #38bdf8;
    padding: 14px;
    font-size: 17px;
}

/* Input Placeholder */
input::placeholder {
    color: #cbd5e1 !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(
        90deg,
        #06b6d4,
        #3b82f6,
        #8b5cf6
    );
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 14px;
    border: none;
    padding: 14px;
    transition: 0.3s;
}

/* Button Hover */
.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0px 0px 20px rgba(59,130,246,0.6);
}

/* Chunk Counter */
.chunk-box {
    background: linear-gradient(
        90deg,
        #0ea5e9,
        #6366f1
    );
    padding: 14px;
    border-radius: 14px;
    text-align: center;
    color: white;
    font-weight: bold;
    font-size: 18px;
    margin-top: 20px;
    margin-bottom: 20px;
}

/* Response Box */
.response-box {
    background: rgba(255,255,255,0.05);
    padding: 30px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
    margin-top: 20px;
    color: white;
    line-height: 1.9;
    font-size: 18px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617;
}

/* Success Message */
.stAlert {
    border-radius: 15px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: #3b82f6;
    border-radius: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)

# ========================= HERO SECTION =========================

st.markdown("""
<div class="main-title">
    🤖 AI RAG PDF Assistant
</div>

<div class="sub-text">
    Upload PDFs • Retrieve Knowledge • Ask Questions • Powered by Groq + LangChain
</div>
""", unsafe_allow_html=True)

# ========================= LOAD ENV =========================

load_dotenv()

# ========================= API KEY =========================

groq_api_key = os.getenv("GROQ_API_KEY")

# ========================= CREATE LLM =========================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

# ========================= FILE UPLOAD =========================

uploaded_file = st.file_uploader(
    "📂 Upload your PDF",
    type="pdf"
)

# ========================= PROCESS PDF =========================

if uploaded_file is not None:

    st.success("✅ PDF Uploaded Successfully")

    # Save temporary PDF
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())

        temp_pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(temp_pdf_path)

    documents = loader.load()

    # ========================= TEXT SPLITTING =========================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    # ========================= CHUNK DISPLAY =========================

    st.markdown(
        f'''
        <div class="chunk-box">
            📚 Total Chunks Created: {len(chunks)}
        </div>
        ''',
        unsafe_allow_html=True
    )

    # ========================= EMBEDDINGS =========================

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ========================= VECTOR DATABASE =========================

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # ========================= RETRIEVER =========================

    retriever = vectorstore.as_retriever()

    # ========================= QUESTION INPUT =========================

    question = st.text_input(
        "💬 Ask a question from the PDF"
    )

    # ========================= RESPONSE BUTTON =========================

    if st.button("🚀 Generate Response"):

        if question:

            with st.spinner("Thinking... 🤔"):

                # Retrieve relevant chunks
                retrieved_docs = retriever.invoke(question)

                # Combine context
                context = "\n\n".join(
                    [doc.page_content for doc in retrieved_docs]
                )

                # ========================= PROMPT =========================

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

                # ========================= GENERATE RESPONSE =========================

                response = llm.invoke(final_prompt)

                # ========================= RESPONSE TITLE =========================

                st.markdown("""
                <h2 style='color:#38bdf8; margin-top:30px;'>
                    🤖 RAG Response
                </h2>
                """, unsafe_allow_html=True)

                # ========================= RESPONSE BOX =========================

                st.markdown(
                    f'''
                    <div class="response-box">
                        {response.content}
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

# ========================= FOOTER =========================

st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit, LangChain, Groq, ChromaDB & HuggingFace Embeddings
</div>
""", unsafe_allow_html=True)
