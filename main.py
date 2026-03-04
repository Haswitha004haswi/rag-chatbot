# main.py - FastAPI + RAG + Web Fallback

import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime

from domain_classifier import classify_query
from db_utils import search_vector_db
from reranker import rerank_chunks
from llm_utils import generate_answer, answer_from_web
from tavily import TavilyClient
from rasa_greetings import rasa_greeting
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ================= LOAD ENV =================
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ================= DOMAIN MEMORY =================
chat_memory = {"education_pdfs": [], "healthcare_pdfs": []}
DOMAIN_MAP = {"Education": "education_pdfs", "Healthcare": "healthcare_pdfs"}
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ================= Pydantic Model =================
class QueryRequest(BaseModel):
    query: str

# ================= MEMORY FUNCTIONS =================
def update_memory(domain, query, answer, max_len=10):
    chat_memory[domain].append(f"User: {query}")
    chat_memory[domain].append(f"Bot: {answer}")
    if len(chat_memory[domain]) > max_len:
        del chat_memory[domain][:len(chat_memory[domain]) - max_len]

def compute_similarity(query1, query2):
    vecs = embedding_model.encode([query1, query2])
    return cosine_similarity([vecs[0]], [vecs[1]])[0][0]

def get_relevant_history(domain, query, n=3, threshold=0.5):
    history = chat_memory.get(domain, [])
    relevant_history = ""
    count = 0
    for i in range(len(history)-2, -1, -2):
        past_query = history[i].replace("User: ", "")
        past_answer = history[i+1].replace("Bot: ", "")
        sim_score = compute_similarity(query, past_query)
        if sim_score >= threshold:
            relevant_history += f"{past_query}\n{past_answer}\n"
            count += 1
            if count >= n: break
    if relevant_history:
        prompt = f"""
Summarize the following related chat history in 1-2 sentences:

Chat History:
{relevant_history}

Summary:
"""
        relevant_history = generate_answer(prompt, "", max_tokens=100)
    return relevant_history

def rewrite_query_with_memory(query):
    if len(query.strip().split()) <= 3: 
        return query
    full_history = ""
    for domain in chat_memory:
        summary = get_relevant_history(domain, query, n=2, threshold=0.5)
        if summary: 
            full_history += summary + "\n"
    if not full_history.strip(): 
        return query
    prompt = f"""
Rewrite the user query into a complete standalone question.

Chat History:
{full_history}

User Query:
{query}

Rewritten Query:
"""
    return generate_answer(prompt, "", max_tokens=150).strip()

def build_smart_context(chunks, max_chars=1000):
    context = ""
    for chunk in chunks:
        if len(context) + len(chunk) > max_chars: break
        context += chunk + "\n\n"
    return context

def tavily_search(query, max_results=3):
    try:
        response = tavily.search(query, max_results=max_results)
        results = [r.get("content", "") for r in response.get("results", [])]
        return "\n".join(results) if results else ""
    except Exception as e:
        print("Tavily search failed:", e)
        return ""

def should_fallback(similarity_scores, threshold=0.4):
    if not similarity_scores: 
        return True
    top_score = max(similarity_scores)
    return top_score < threshold

# ================= ASK QUESTION =================
def ask_question(query, top_k=6, db_sim_threshold=0.4):
    # Greeting check
    greeting_response = rasa_greeting(query)
    if greeting_response: 
        return {"answer": greeting_response, "source": "Greeting", "domain": None}

    # Rewrite query with memory
    rewritten_query = rewrite_query_with_memory(query)

    # Classify domain
    domain_result = classify_query(rewritten_query)
    if domain_result == "Other":
        return {"answer": "🙂 Sorry! I only answer medical education and healthcare questions.", "source": None, "domain": None}

    domain_db = DOMAIN_MAP[domain_result]
    relevant_history = get_relevant_history(domain_db, rewritten_query, n=5, threshold=0.5)

    # Retrieve from vector DB
    retrieved_chunks, similarity_scores = search_vector_db(
        query=rewritten_query, domain=domain_db, top_k=top_k, return_scores=True
    )
    reranked_chunks = rerank_chunks(rewritten_query, retrieved_chunks, top_k=top_k) if retrieved_chunks else []

    # --- Vector DB response ---
    if reranked_chunks and not should_fallback(similarity_scores, db_sim_threshold):
        source = f"{domain_result} PDFs"
        context = build_smart_context(reranked_chunks)
        prompt = f"""
Chat History:
{relevant_history}

Context:
{context}

Question:
{rewritten_query}

Answer clearly:
"""
        answer = generate_answer(prompt, context, max_tokens=400)
    # --- Fallback to web search ---
    else:
        web_context = tavily_search(rewritten_query)
        if web_context:
            prompt = f"""
Chat History:
{relevant_history}

Web Context:
{web_context}

Question:
{rewritten_query}

Answer clearly:
"""
            answer = answer_from_web(prompt, web_context, max_tokens=400)
            source = "Web Search"
        else:
            answer = "No relevant information found."
            source = "No Source"

    update_memory(domain_db, rewritten_query, answer)

    return {"answer": answer, "source": source, "domain": domain_result}

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
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)