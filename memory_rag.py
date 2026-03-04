# memory_rag.py (FINAL — safe for free accounts)

from domain_classifier import classify_query
from db_utils import search_vector_db
from reranker import rerank_chunks
from llm_utils import generate_answer

# ==============================
# DOMAIN-SPECIFIC MEMORY
# ==============================
chat_memory = {
    "education_pdfs": [],
    "healthcare_pdfs": []
}

# ==============================
# DOMAIN MAP
# ==============================
DOMAIN_MAP = {
    "Education": "education_pdfs",
    "Healthcare": "healthcare_pdfs"
}

# ==============================
# UPDATE MEMORY
# ==============================
def update_memory(domain, query, answer, max_len=10):
    chat_memory[domain].append(f"User: {query}")
    chat_memory[domain].append(f"Bot: {answer}")

    # Limit memory size
    if len(chat_memory[domain]) > max_len:
        del chat_memory[domain][:len(chat_memory[domain]) - max_len]

# ==============================
# MEMORY SUMMARIZATION
# ==============================
def get_recent_history(domain, n=3, summarize=True):
    history = chat_memory.get(domain, [])
    recent_history = "\n".join(history[-n:])

    if summarize and recent_history:
        # Summarize chat memory to 1-2 sentences
        prompt = f"""
Summarize the following chat history in 1-2 sentences, keeping only key topics:

Chat History:
{recent_history}

Summary:
"""
        recent_history = generate_answer(prompt, "")
    
    return recent_history

# ==============================
# QUERY REWRITING
# ==============================
def rewrite_query_with_memory(query):
    full_history = ""
    for d in chat_memory:
        full_history += "\n".join(chat_memory[d][-2:]) + "\n"

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
    rewritten = generate_answer(prompt, "")
    return rewritten.strip()

# ==============================
# CONTEXT BUILDER
# ==============================
def build_context(chunks, max_chars=1000):  # truncated for token safety
    context = ""
    for chunk in chunks:
        if len(context) + len(chunk) > max_chars:
            break
        context += chunk + "\n\n"
    return context

# ==============================
# MEMORY RAG FUNCTION
# ==============================
def memory_rag(query, top_k=3):

    print(f"\n🟢 User Query: {query}")

    # 🔁 Rewrite query using memory
    rewritten_query = rewrite_query_with_memory(query)
    print(f"🔁 Rewritten Query: {rewritten_query}")

    # 1️⃣ Domain classification
    domain_result = classify_query(rewritten_query)
    print(f"📂 Classified Domain: {domain_result}")

    if domain_result == "Other":
        return "🙂 This bot only handles medical education and healthcare."

    domain_db = DOMAIN_MAP[domain_result]

    # 2️⃣ Load domain memory (summarized)
    history_text = get_recent_history(domain_db, n=5, summarize=True)

    # 3️⃣ Retrieve from DB
    chunks = search_vector_db(
        query=rewritten_query,
        domain=domain_db,
        top_k=top_k
    )

    if not chunks:
        return "❌ No relevant data found in database."

    # 4️⃣ Rerank chunks
    reranked_chunks = rerank_chunks(rewritten_query, chunks, top_k=top_k)

    # 5️⃣ Build context (truncated for tokens)
    context = build_context(reranked_chunks, max_chars=1000)

    # 6️⃣ Prepare prompt
    prompt = f"""
You are a domain-aware assistant.

Chat History:
{history_text}

Context:
{context}

User Question:
{rewritten_query}

Answer clearly and accurately:
"""

    # 7️⃣ Generate answer
    answer = generate_answer(prompt, context)

    # 8️⃣ Update memory
    update_memory(domain_db, rewritten_query, answer)

    return answer

# ==============================
# TEST LOOP
# ==============================
if __name__ == "__main__":
    print("===== MEMORY RAG CHATBOT (FINAL WITH SUMMARIZATION) =====")

    while True:
        q = input("\nAsk: ")

        if q.lower() == "exit":
            print("👋 Exiting...")
            break

        response = memory_rag(q)
        print("\n💬", response)