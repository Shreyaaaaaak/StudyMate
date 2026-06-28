from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Step 1 — Load the embedding model (same one we used to store)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Step 2 — Connect to existing ChromaDB (don't rebuild, just load)
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Step 3 — Ask a question
question = "What is the difference between TCP and UDP?"

# Step 4 — Retrieve top 3 most relevant chunks
results = vectorstore.similarity_search(question, k=3)

print(f"Question: {question}")
print(f"\nTop {len(results)} relevant chunks from your notes:\n")

for i, doc in enumerate(results):
    print(f"--- Chunk {i+1} (Page {doc.metadata.get('page', '?') + 1}) ---")
    print(doc.page_content)
    print()