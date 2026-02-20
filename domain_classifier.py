import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

# ==============================
# LOAD EMBEDDING MODEL
# ==============================
model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================
# DOMAIN TEXTS (for embedding-based similarity)
# ==============================
DOMAIN_TEXTS = {
    "Education": [
        "medical education, anatomy, physiology, pathology, pharmacology, clinical skills, medical exams, lecture notes",
        "medical study materials, tutorials, case studies, diagnostic procedures"
    ],
    "Healthcare": [
        "healthcare, patient care, disease, diagnosis, treatment, hospital, clinical practice, public health, epidemiology, preventive medicine",
        "medical record documentation, hospital guidelines, patient management, healthcare policy"
    ]
}

# Precompute embeddings for each domain
DOMAIN_INDEX = {
    domain: model.encode(texts, normalize_embeddings=True)
    for domain, texts in DOMAIN_TEXTS.items()
}

# ==============================
# LLM CLASSIFICATION HELPER
# ==============================
def llm_classify(query):
    """
    LLM fallback to strictly classify the query into Education, Healthcare, or Other
    """
    prompt = f"""
You are a strict medical domain expert.

Classify this query:

- Education: strictly medical education (study materials, anatomy, physiology, pharmacology, pathology, clinical exams)
- Healthcare: patient care, diseases, diagnosis, treatment, hospital practice, public health, community health, epidemiology, preventive medicine, hospital guidelines
- Other: not medical or outside medical domains

Return ONLY: Education, Healthcare, or Other.

Query: "{query}"
"""
    response = openai.ChatCompletion.create(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=50
    )
    domain = response["choices"][0]["message"]["content"].strip().title()
    if domain not in ["Education", "Healthcare", "Other"]:
        return "Other"
    return domain

# ==============================
# HYBRID MEDICAL DOMAIN CLASSIFIER
# ==============================
def classify_medical_domain(query: str,
                            emb_threshold=0.2,
                            health_emb_fallback=0.45):
    """
    Hybrid classifier:
    1. Use embeddings first.
    2. If embeddings are weak, fallback to LLM classification.
    """
    query_emb = model.encode(query, normalize_embeddings=True)

    # Compute embedding similarity per domain
    scores = {}
    for domain, embeddings in DOMAIN_INDEX.items():
        sims = np.dot(embeddings, query_emb)
        scores[domain] = float(np.max(sims))

    edu_score = scores["Education"]
    health_score = scores["Healthcare"]

    print(f"\nEmbedding similarity → Education: {edu_score:.3f}, Healthcare: {health_score:.3f}")

    # If strong embedding exists → use it
    if edu_score >= emb_threshold:
        print("Embedding indicates potential Education query → confirming with LLM")
        llm_result = llm_classify(query)
        if llm_result == "Education":
            print("LLM confirmed → Education")
            return "Education"

    if health_score >= health_emb_fallback:
        print("Strong Healthcare embedding → classify as Healthcare")
        return "Healthcare"

    # If embeddings weak → fallback to LLM
    print("Embeddings weak → using LLM for final classification")
    return llm_classify(query)

# ==============================
# TEST MODE
# ==============================
if __name__ == "__main__":
    while True:
        q = input("\nEnter query: ")
        if q.lower() == "exit":
            break
        result = classify_medical_domain(q)
        print(f"📂 Classified Domain: {result}")