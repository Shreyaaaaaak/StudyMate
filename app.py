import os
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
st.caption("Ask anything from your KIIT notes — answers grounded in your own material.")

@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Build vector store if it doesn't exist
    if not os.path.exists("chroma_db"):
        with st.spinner("Building knowledge base from your notes... (first run only)"):
            loader = PyPDFLoader("notes.pdf")
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = splitter.split_documents(pages)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )
    else:
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

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

chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask something from your notes..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your notes..."):
            answer = chain.invoke(question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})