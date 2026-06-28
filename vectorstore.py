from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Step 1 — Load and chunk (same as before)
loader = PyPDFLoader("notes.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(pages)

# Step 2 — Load the embedding model
print("Loading embedding model... (first run downloads ~90MB, be patient)")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Step 3 — Store chunks as vectors in ChromaDB
print("Embedding chunks and storing in ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print(f"Done! {vectorstore._collection.count()} chunks stored in ChromaDB")