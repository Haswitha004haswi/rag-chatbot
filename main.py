# main.py

import os
from dotenv import load_dotenv
from domain_classifier import classify_medical_domain
from db_utils import search_vector_db
from reranker import rerank_chunks
from llm_utils import generate_answer, answer_from_web
from retriever_evaluator import precision_at_k, recall_at_k, mrr_score
from tavily import TavilyClient

# ==============================
# LOAD ENV VARIABLES
# ==============================
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ==============================
# DOMAIN → VECTOR DB MAP
# ==============================
DOMAIN_MAP = {
    "Education": "education_pdfs",
    "Healthcare": "healthcare_pdfs"
}

# ==============================
# SMART CONTEXT BUILDER
# ==============================
def build_smart_context(chunks, max_chars=3000):
    context = ""
    used = 0

    for chunk in chunks:
        if len(context) + len(chunk) > max_chars:
            break
        context += chunk + "\n\n"
        used += 1

    print(f"🧠 Smart Context → Using {used} chunks")
    return context

# ==============================
# WEB SEARCH (TAVILY)
# ==============================
def tavily_search(query, max_results=3):
    try:
        response = tavily.search(query, max_results=max_results)
        results = [r.get("content", "") for r in response.get("results", [])]
        return "\n".join(results) if results else ""
    except Exception as e:
        print("⚠️ Tavily search failed:", e)
        return ""

# ==============================
# RETRIEVAL CONFIDENCE CHECK
# ==============================
def should_fallback(similarity_scores, threshold=0.4):
    if not similarity_scores:
        return True
    return max(similarity_scores) < threshold

# ==============================
# MAIN PIPELINE
# ==============================
def ask_question(query, top_k=6, db_sim_threshold=0.4):

    # 1️⃣ DOMAIN CLASSIFICATION
    domain_result = classify_medical_domain(query)
    print(f"\n📂 Classified Domain: {domain_result}")

    if domain_result == "Other":
        return "🙂 This assistant only answers medical education and healthcare questions."

    domain_db = DOMAIN_MAP[domain_result]

    # 2️⃣ SEARCH VECTOR DB
    retrieved_chunks, similarity_scores = search_vector_db(
        query=query,
        domain=domain_db,
        top_k=top_k,
        return_scores=True
    )

    print(f"🔎 Searching DB: {domain_db}")
    print(f"📄 Retrieved chunks: {len(retrieved_chunks)}")

    # 3️⃣ RERANK
    reranked_chunks = rerank_chunks(query, retrieved_chunks, top_k=top_k) if retrieved_chunks else []

    # 4️⃣ INITIALIZE
    metrics = {}
    answer = ""
    source = ""

    # ==============================
    # VECTOR DB DECISION
    # ==============================
    if reranked_chunks and not should_fallback(similarity_scores, db_sim_threshold):
        print("✅ High retrieval confidence → Using Vector DB")
        source = "Vector DB"

        # Build context and generate answer
        context = build_smart_context(reranked_chunks)
        answer = generate_answer(query, context)

        # Compute retriever metrics ONLY for Vector DB
        metrics = {
            "Precision@k": precision_at_k(query, reranked_chunks, k=top_k, threshold=db_sim_threshold),
            "Recall@k": recall_at_k(query, reranked_chunks, k=top_k, threshold=db_sim_threshold),
            "MRR": mrr_score(query, reranked_chunks, threshold=db_sim_threshold)
        }

        # Optional second-level fallback if DB answer is empty
        if answer.strip().lower() == "i could not find this in uploaded pdfs.":
            print("⚠️ Answer not found in DB → Falling back to Web Search")
            web_context = tavily_search(query)
            if web_context:
                answer = answer_from_web(query, web_context)
                source = "Web Search"
            else:
                answer = "❌ I could not find any relevant information online."
                source = "No Source"

    else:
        # ==============================
        # FALLBACK TO WEB SEARCH
        # ==============================
        print("🌐 Low retrieval confidence → Using Web Search")
        web_context = tavily_search(query)
        if web_context:
            answer = answer_from_web(query, web_context)
            source = "Web Search"
        else:
            answer = "❌ I could not find any relevant information online."
            source = "No Source"

    # ==============================
    # OUTPUT FORMAT
    # ==============================
    output = f"""
----------------------------
✅ SOURCE USED: {domain_db if source == 'Vector DB' else source}

💬 ANSWER:
{answer}
"""

    # Show metrics only if Vector DB was used
    if source == "Vector DB" and metrics:
        output += f"""
📊 Retriever Metrics (DB only):
Precision@k: {metrics['Precision@k']:.2f}
Recall@k: {metrics['Recall@k']:.2f}
MRR: {metrics['MRR']:.2f}
"""

    output += "----------------------------"
    return output

# ==============================
# INTERACTIVE MODE
# ==============================
if __name__ == "__main__":
    print("\n===== MEDICAL RAG ASSISTANT =====")

    while True:
        q = input("\nAsk: ")

        if q.lower() == "exit":
            print("👋 Goodbye!")
            break

        result = ask_question(q)
        print(result)
