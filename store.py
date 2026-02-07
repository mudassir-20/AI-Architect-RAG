import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load environment variables
load_dotenv()

# 2. Load and Split the PDF
print("--- Loading and splitting PDF ---")
loader = PyPDFLoader("data.pdf")
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)

# 3. Setup the Embedding Model (The "Translator")
print(f"--- Creating embeddings for {len(chunks)} chunks ---")
# This model is small, fast, and runs for free in your Codespace
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Create the "Filing Cabinet" (Vector Store)
print("--- Saving to Vector Database (db_storage) ---")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./db_storage"
)

print("✅ Success! You now have a searchable 'AI Memory' in the db_storage folder.")