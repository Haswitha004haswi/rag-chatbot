# domain_classifier.py
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import dotenv

# ==============================
# LOAD ENV & OPENAI SETUP
# ==============================
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

# ==============================
# EMBEDDING MODEL
# ==============================
model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================
# DOMAIN REFERENCE TEXTS
# ==============================
DOMAIN_TEXTS = {
    "Education": [
        "medical education, anatomy, physiology, pharmacology, pathology",
        "medical lectures, MBBS subjects, NEET PG preparation",
        "clinical theory, textbooks, medical concepts explanation"
    ],
    "Healthcare": [
        "patient care, diagnosis, treatment, disease management",
        "hospital practice, doctors, prescriptions, surgery",
        "public health, epidemiology, prevention, healthcare systems",
        "anatomy, organ functions, physiology, symptoms, brain, heart, liver"
    ]
}

# Precompute embeddings
DOMAIN_INDEX = {
    domain: model.encode(texts, normalize_embeddings=True)
    for domain, texts in DOMAIN_TEXTS.items()
}

# ==============================
# STEP 1 → MEDICAL FILTER
# ==============================
MEDICAL_KEYWORDS = [
    "anatomy", "physiology", "brain", "heart", "lungs", "organ", 
    "disease", "symptom", "treatment", "patient", "medicine", "hospital"
]

def is_medical_query(query: str) -> bool:
    """Return True if query is medical-related."""
    query_lower = query.lower()
    if any(word in query_lower for word in MEDICAL_KEYWORDS):
        return True

    # LLM strict classifier fallback
    prompt = f"""
You are a medical classifier.

Rules:
- Return YES only if query is about medicine, health, anatomy, or organ functions
- Return NO if unrelated to medicine or health

Query: "{query}"
Answer ONLY: YES or NO
"""
    response = openai.ChatCompletion.create(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5
    )
    result = response["choices"][0]["message"]["content"].strip().lower()
    return "yes" in result

# ==============================
# STEP 2 → EMBEDDING-BASED CLASSIFICATION
# ==============================
def embedding_classify(query: str) -> str:
    query_emb = model.encode(query, normalize_embeddings=True)
    scores = {}
    for domain, embeddings in DOMAIN_INDEX.items():
        sims = np.dot(embeddings, query_emb)
        scores[domain] = float(np.max(sims))
    # Debug
    print(f"Embedding similarity → Education: {scores['Education']:.3f}, Healthcare: {scores['Healthcare']:.3f}")
    return "Education" if scores["Education"] > scores["Healthcare"] else "Healthcare"

# ==============================
# STEP 3 → LLM CONFIRMATION
# ==============================
def llm_subclassify(query: str) -> str:
    prompt = f"""
You are a medical expert.

Classify the query into:
- Education → theory, definitions, explanations
- Healthcare → diagnosis, treatment, patient care, anatomy, organ functions

Return ONLY: Education or Healthcare

Query: "{query}"
"""
    response = openai.ChatCompletion.create(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )
    result = response["choices"][0]["message"]["content"].strip().title()
    if result not in ["Education", "Healthcare"]:
        return "Education"
    return result

# ==============================
# FINAL CLASSIFIER
# ==============================
def classify_query(query: str) -> str:
    print(f"\nQuery: {query}")

    # Step 1: Check medical relevance
    if not is_medical_query(query):
        print("❌ Not a medical query")
        return "Other"
    print("✅ Medical query detected")

    # Step 2: Embedding similarity
    emb_result = embedding_classify(query)

    # Step 3: LLM confirmation
    llm_result = llm_subclassify(query)
    print(f"Embedding → {emb_result}, LLM → {llm_result}")

    return llm_result

# ==============================
# MAIN LOOP FOR TESTING
# ==============================
if __name__ == "__main__":
    while True:
        query = input("\nEnter query: ")
        if query.lower() == "exit":
            break

        domain = classify_query(query)
        if domain == "Other":
            print("⚠️ I only answer medical-related questions. Please ask a medical query.")
        else:
            print(f"📂 Domain: {domain}")