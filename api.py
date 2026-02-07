from fastapi import FastAPI
from pydantic import BaseModel

# Import your existing function
from query import ask_question

app = FastAPI(
    title="The Architect – Enterprise RAG Agent",
    description="Enterprise RAG Agent with Self-Correction",
    version="1.0"
)

# Request schema
class QuestionRequest(BaseModel):
    question: str

# Response schema
class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AnswerResponse)
def ask_agent(request: QuestionRequest):
    answer = ask_question(request.question)
    return {"answer": answer}
