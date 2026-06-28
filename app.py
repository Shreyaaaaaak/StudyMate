import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(
    page_title="StudyMate",
    page_icon="📚",
    layout="centered"
)

st.title("📚 StudyMate")
st.caption("Upload your notes, ask anything — answers grounded in your own material.")

# Load embedding model once — cached globally
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load LLM once — cached globally
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

embeddings = load_embeddings()
llm = load_llm()

# PDF uploader
uploaded_files = st.file_uploader(
    "Upload your notes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

# Build vector store from uploaded PDFs
def build_vectorstore(files):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    for file in files:
        # Save uploaded file to a temp location so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
        os.unlink(tmp_path)  # clean up temp file

    # In-memory ChromaDB — no persist_directory
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings
    )
    return vectorstore

# Build chain from vectorstore
def build_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate.from_template("""You are StudyMate, a helpful study assistant for KIIT students.
Use ONLY the following context from the student's notes to answer the question.
If the answer is not in the context, say "I couldn't find this in your notes."
Keep answers clear and structured.

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Process uploaded PDFs
if uploaded_files:
    # Check if new files were uploaded
    file_names = [f.name for f in uploaded_files]
    if file_names != st.session_state.processed_files:
        with st.spinner(f"Processing {len(uploaded_files)} PDF(s)... please wait"):
            vectorstore = build_vectorstore(uploaded_files)
            st.session_state.chain = build_chain(vectorstore)
            st.session_state.processed_files = file_names
            st.session_state.messages = []  # reset chat on new upload
        st.success(f"✅ Ready! Loaded {len(uploaded_files)} file(s): {', '.join(file_names)}")

# Chat UI
if st.session_state.chain is None:
    st.info("👆 Upload your PDF notes above to get started.")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if question := st.chat_input("Ask something from your notes..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your notes..."):
                answer = st.session_state.chain.invoke(question)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})