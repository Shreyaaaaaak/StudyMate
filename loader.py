from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1 — Load the PDF
loader = PyPDFLoader("notes.pdf")
pages = loader.load()

print(f"Total pages loaded: {len(pages)}")
print(f"\nFirst 500 characters of page 1:\n{pages[0].page_content[:500]}")

# Step 2 — Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(pages)

print(f"\nTotal chunks created: {len(chunks)}")
print(f"\nExample chunk:\n{chunks[0].page_content}")