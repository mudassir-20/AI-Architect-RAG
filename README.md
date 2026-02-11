# 🏗️ The AI Architect-RAG

# Enterprise RAG Agent with Self-Correction

The Architect is an enterprise-grade Retrieval-Augmented Generation (RAG) system designed to answer questions from large documents reliably, safely, and transparently.

Unlike basic RAG demos, this project implements multi-agent self-correction, source-aware answers, and automatic web fallback, closely mirroring how production AI systems are built in real companies.

## 🚀 Key Features

**📄 Document-based Question Answering**

Answers questions strictly from a provided PDF (NVIDIA Annual Report) using semantic search.

**🧠 Multi-Query Retrieval**

Automatically rewrites user queries into multiple semantic variants to improve recall.

**🔁 Multi-Agent Self-Correction Pipeline**

* Answer Agent – generates an initial response
* Critic Agent – evaluates correctness, grounding, and clarity
* Refiner Agent – improves the answer based on critic feedback

**🛑 Hallucination Control**

* Verifies whether answers are supported by document context
* Responds with “I don’t know based on the document” when appropriate

**🌐 Web Search Fallback**

* Automatically detects external questions
* Uses web search (Tavily API) when the document does not contain the answer
* Clearly separates document-based vs web-based answers

**🧾 Source-Aware Responses**

* Adds page-level citations for document answers
* Never attaches document sources to web-based answers

**💬 Conversational Memory**

Maintains short-term chat history for follow-up questions

**⚙️ FastAPI Deployment**

* Exposes the agent via a clean /ask API endpoint
* Ready for frontend integration or production deployment

## 🧠 System Architecture

User Query
 ➡️
Query Rewriter (Multi-query retrieval)
   ➡️
Vector Database (ChromaDB + PDF embeddings)
   ➡️
Answer Agent (LLM)
   ➡️
Critic Agent (quality & grounding check)
   ➡️
Refiner Agent (self-correction)
   ➡️
Verifier (hallucination guard)

   ⬇️
   
Decision Router

   ├─ Document Answer + Page Sources
   
   └─ Web Search Fallback + Web Answer
   
 ⬇️
 
Final Response + Memory

## 🛠️ Tech Stack

* LLM: Groq (LLaMA 3.x)

* Embeddings: Sentence-Transformers (MiniLM)
* Vector Database: ChromaDB

* Web Search: Tavily API

* Backend: FastAPI

* Language: Python

* Deployment Environment: GitHub Codespaces

## 📂 Project Structure

├── api.py          # FastAPI application

├── query.py        # Core RAG + agent logic

├── ingest.py       # PDF ingestion & embedding

├── store.py        # Vector store utilities

├── data.pdf        # Source document (NVIDIA Annual Report)

├── README.md       # Project documentation

└── .gitignore

## 🎯 Why This Project Matters

Most RAG examples stop at retrieval + generation.

This project goes further by adding:

* Self-correction

* Verification

* Provenance control

* Web fallback decision logic

### 🔍 Example Queries

What is NVIDIA Omniverse?

Explain it in simple terms

Who is the CEO of NVIDIA in 2025?

The system will automatically decide whether to answer from the document or use web search.

### 👤 Author

Mudassir Ansari

Computer Science (AI & ML)

Aspiring AI Engineer




