import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from tavily import TavilyClient
# from langchain_openai import ChatOpenAI # Uncomment if using OpenAI 
# --- Conversation Memory ---
chat_history = []
MAX_HISTORY = 4  # keep last 4 question-answer pairs

load_dotenv()

# 1. Load the "Filing Cabinet" we built in the last step
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./db_storage", embedding_function=embeddings)

# 2. Setup the LLM (The "Brain")
# If using OpenAI, change this to: llm = ChatOpenAI(model="gpt-4o")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


def generate_search_queries(question):
    prompt = f"""
You are an AI assistant helping improve document retrieval.

Generate 3 different search queries that could help find
relevant information in a document for the question below.

Return each query on a new line.
Do NOT add numbering or extra text.

Question:
{question}
"""
    response = llm.invoke(prompt).content.strip()
    queries = [q.strip() for q in response.split("\n") if q.strip()]
    return queries[:3]


# 3. The Retrieval Logic
def ask_question(question):
    # -----------------------------
    # 0. Init flags & constants
    # -----------------------------
    answer_from_document = True
    external_triggers = [
        "who is", "ceo", "founder", "current",
        "latest", "today", "2024", "2025"
    ]

    print(f"\n--- Searching for: {question} ---")

    # -----------------------------
    # 1. Multi-query retrieval
    # -----------------------------
    search_queries = generate_search_queries(question)

    docs = []
    for q in search_queries:
        docs.extend(vector_db.similarity_search(q, k=2))

    # -----------------------------
    # 2. Build context + sources
    # -----------------------------
    context_chunks = []
    sources = set()

    for d in docs:
        context_chunks.append(d.page_content)
        if "page" in d.metadata:
            sources.add(f"Page {d.metadata['page']}")

    context = "\n\n".join(context_chunks)

    # -----------------------------
    # 3. Conversation memory
    # -----------------------------
    history_text = ""
    for q, a in chat_history[-MAX_HISTORY:]:
        history_text += f"User: {q}\nAssistant: {a}\n"

    # -----------------------------
    # 4. Prompt
    # -----------------------------
    prompt = f"""
You are an enterprise AI assistant answering questions strictly
from NVIDIA's Annual Report.

Use ONLY the context provided below.
If the answer is not present, say: "I don't know based on the document."

Conversation History:
{history_text}

Context:
{context}

Question:
{question}
"""

    # -----------------------------
    # 5. Answer Agent
    # -----------------------------
    answer = llm.invoke(prompt).content.strip()

    # -----------------------------
    # 6. Self-correction loop
    # -----------------------------
    MAX_REFINES = 2
    for _ in range(MAX_REFINES):
        critique = critic_agent(answer, context)
        if critique.strip().upper() == "OK":
            break

        refined = refiner_agent(answer, critique, context)
        if refined == answer:
            break

        answer = refined

    # -----------------------------
    # 7. Final validation & routing
    # -----------------------------
    is_valid = verify_answer(answer, context)

    if not is_valid:
        is_external = any(t in question.lower() for t in external_triggers)

        if is_external:
            web_answer = web_search_fallback(question)
            answer = (
                "⚠️ Not found in the document.\n"
                "🌐 Web-based answer:\n"
                f"{web_answer}"
            )
            answer_from_document = False
        else:
            answer = "I don't know based on the document."

    # -----------------------------
    # 8. Append sources (ONLY if doc answer)
    # -----------------------------
    if (
        answer_from_document
        and sources
        and not answer.lower().startswith("i don't know")
    ):
        answer = f"{answer}\n\nSources:\n" + "\n".join(sorted(sources))

    # -----------------------------
    # 9. Save memory
    # -----------------------------
    chat_history.append((question, answer))
    return answer



def verify_answer(answer, context):
    prompt = f"""
You are a strict AI verifier.

Check whether the answer below is fully supported by the given context.
Reply with ONLY one word: YES or NO.

Context:
{context}

Answer:
{answer}
"""
    verdict = llm.invoke(prompt).content.strip().upper()
    return verdict.startswith("YES")

def critic_agent(answer, context):
    prompt = f"""
You are a strict enterprise AI critic.

Your job is to review the answer and decide if it is:
- Fully supported by the context
- Clear and specific
- Free of unsupported claims

Context:
{context}

Answer:
{answer}

Reply with:
- "OK" if the answer is good
- Otherwise, list the problems briefly.
"""
    response = llm.invoke(prompt)
    return response.content.strip()

def refiner_agent(answer, critique, context):
    prompt = f"""
You are an enterprise AI refiner.

Your task:
- Improve the answer using ONLY the provided context
- Address the critic’s feedback
- Be specific and concise
- Do NOT add new facts
- If the answer cannot be improved with the context, return the original answer

Context:
{context}

Original Answer:
{answer}

Critic Feedback:
{critique}

Refined Answer:
"""
    response = llm.invoke(prompt)
    return response.content.strip()



def web_search_fallback(question):
    print("🌐 Falling back to web search...")

    results = tavily_client.search(
        query=question,
        search_depth="basic",
        max_results=5
    )

    web_context = ""
    for r in results.get("results", []):
        web_context += f"- {r['title']}: {r['content']}\n"

    prompt = f"""
You are a factual AI assistant.

Answer the question using ONLY the web search results below.
If the answer cannot be found, say "I don't know."

Web Results:
{web_context}

Question:
{question}
"""
    response = llm.invoke(prompt)
    return response.content.strip()


# 4. Chat Loop (Continuous Conversation)

print("\nRAG Assistant is ready.")
print("Ask questions about your document.")
print("Type 'exit' to quit.\n")

while True:
    user_query = input("You: ").strip()

    if user_query.lower() in ["exit", "quit"]:
        print("Goodbye 👋")
        break

    if not user_query:
        continue

    answer = ask_question(user_query)

    print("\n--- AI ANSWER ---")
    print(answer)
    print("\n" + "=" * 60)



 # ----- TEMP REFINER TEST -----
sample_answer = "NVIDIA Omniverse is a collaboration platform."
sample_context = "NVIDIA Omniverse enables real-time 3D design collaboration and simulation for industrial digital twins."
sample_critique = "The answer is vague and does not mention real-time 3D or digital twins."

print("\n--- REFINER OUTPUT ---")
print(refiner_agent(sample_answer, sample_critique, sample_context))

