import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the secret API key from your .env file
load_dotenv()

# 2. Load the PDF
print("--- Step 1: Loading PDF ---")
loader = PyPDFLoader("data.pdf")
pages = loader.load()

# 3. Chunk the text
# Why? LLMs have a "Context Window" (a limit on how much they can read). 
# We chop the PDF into 1000-character pieces so the AI can find specific parts easily.
print("--- Step 2: Splitting into chunks ---")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
)
chunks = text_splitter.split_documents(pages)

print(f"Done! Your PDF was split into {len(chunks)} small pieces.")
print("\nHere is a preview of Chunk #1:")
print("-" * 30)
print(chunks[0].page_content[:300] + "...")