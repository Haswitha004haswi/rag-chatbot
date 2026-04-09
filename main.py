# main.py - FastAPI + RAG + Web Fallback

import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from domain_classifier import classify_query
from db_utils import search_vector_db
from reranker import rerank_chunks
from llm_utils import generate_answer, answer_from_web
from tavily import TavilyClient
from rasa_greetings import rasa_greeting

# ================= LOAD ENV =================
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ================= DOMAIN MEMORY =================
chat_memory = {"education_pdfs": [], "healthcare_pdfs": []}
DOMAIN_MAP = {"Education": "education_pdfs", "Healthcare": "healthcare_pdfs"}

# ================= Pydantic Model =================
class QueryRequest(BaseModel):
    query: str


# ================= MEMORY FUNCTION =================
def update_memory(domain, query, answer, max_len=10):
    chat_memory[domain].append(f"User: {query}")
    chat_memory[domain].append(f"Bot: {answer}")

    if len(chat_memory[domain]) > max_len:
        del chat_memory[domain][:len(chat_memory[domain]) - max_len]


# ================= TAVILY SEARCH =================
def tavily_search(query, max_results=3):
    try:
        response = tavily.search(query, max_results=max_results)
        results = [r.get("content", "") for r in response.get("results", [])]
        return "\n".join(results) if results else ""
    except Exception as e:
        print("Tavily search failed:", e)
        return ""


# ================= ASK QUESTION =================
def ask_question(query, top_k=6):

    # Greeting check
    greeting_response = rasa_greeting(query)
    if greeting_response:
        return {
            "answer": greeting_response,
            "source": "Greeting",
            "domain": None
        }

    # Domain classification
    domain_result = classify_query(query)

    if domain_result == "Other":
        return {
            "answer": "🙂 Sorry! I only answer Education and Healthcare questions.",
            "source": None,
            "domain": None
        }

    domain_db = DOMAIN_MAP[domain_result]

    # Retrieve from vector DB
    retrieved_chunks, similarity_scores = search_vector_db(
        query=query,
        domain=domain_db,
        top_k=top_k,
        return_scores=True
    )

    reranked_chunks = rerank_chunks(query, retrieved_chunks, top_k=top_k) if retrieved_chunks else []

    # ✅ Use PDF only if similarity is high
    if reranked_chunks and similarity_scores and max(similarity_scores) > 0.4:

        context = "\n\n".join(reranked_chunks)

        prompt = f"""
Context:
{context}

Question:
{query}

Answer clearly:
"""

        answer = generate_answer(prompt, context, max_tokens=400)
        source = f"{domain_result} PDFs"

    else:
        # Web fallback
        web_context = tavily_search(query)

        if web_context:

            prompt = f"""
Web Context:
{web_context}

Question:
{query}

Answer clearly:
"""

            answer = answer_from_web(prompt, web_context, max_tokens=400)
            source = "Web Search"

        else:
            answer = "No relevant information found."
            source = "No Source"

    update_memory(domain_db, query, answer)

    return {
        "answer": answer,
        "source": source,
        "domain": domain_result
    }


# ================= FASTAPI =================
app = FastAPI(title="Medical & Education RAG Assistant")


@app.post("/ask")
def ask(request: QueryRequest):
    return ask_question(request.query)


@app.get("/")
def root():
    return {"message": "Medical & Education RAG Assistant is running"}


# ================= MAIN =================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # ✅ Reads Render's dynamic PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
