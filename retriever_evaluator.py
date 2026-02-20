# retriever_evaluator.py

import numpy as np
from db_utils import search_vector_db,embed_text

# =====================================
# COSINE SIMILARITY
# =====================================
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# =====================================
# AUTOMATIC RELEVANCE CHECK
# =====================================
def is_relevant(query, chunk, threshold=0.55):
    """
    Automatically determine if retrieved chunk is relevant
    using embedding similarity with the query.
    """
    q_emb = embed_text(query)
    c_emb = embed_text(chunk)
    score = cosine_similarity(q_emb, c_emb)
    return score >= threshold, score

# =====================================
# METRIC HELPERS
# =====================================
def precision_at_k(query, retrieved_chunks, k=5, threshold=0.55):
    if not retrieved_chunks:
        return 0.0
    relevant_count = sum(is_relevant(query, chunk, threshold)[0] for chunk in retrieved_chunks[:k])
    return relevant_count / k

def recall_at_k(query, retrieved_chunks, k=5, threshold=0.55):
    if not retrieved_chunks:
        return 0.0
    # Treat all retrieved chunks above threshold as "relevant" in this automatic evaluation
    relevant_count = sum(is_relevant(query, chunk, threshold)[0] for chunk in retrieved_chunks[:k])
    # Maximum possible relevant = number of retrieved chunks above threshold
    return relevant_count / min(k, len(retrieved_chunks))

def mrr_score(query, retrieved_chunks, threshold=0.55):
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        relevant, _ = is_relevant(query, chunk, threshold)
        if relevant:
            return 1 / rank
    return 0.0

# =====================================
# EVALUATE RETRIEVER AUTOMATICALLY
# =====================================
def evaluate_retriever(test_queries, domain, top_k=6, threshold=0.55):
    """
    Evaluate retriever automatically for any query using embedding similarity.
    Prints only overall metrics.
    """
    precisions, recalls, rranks = [], [], []

    for query in test_queries:
        retrieved_chunks = search_vector_db(query=query, domain=domain, top_k=top_k)
        precisions.append(precision_at_k(query, retrieved_chunks, k=top_k, threshold=threshold))
        recalls.append(recall_at_k(query, retrieved_chunks, k=top_k, threshold=threshold))
        rranks.append(mrr_score(query, retrieved_chunks, threshold=threshold))

    metrics = {
        "Precision@k": float(np.mean(precisions)),
        "Recall@k": float(np.mean(recalls)),
        "MRR": float(np.mean(rranks))
    }

    print("\n========== Overall Retriever Metrics ==========")
    print(metrics)
    return metrics

# =====================================
# Example usage
# =====================================
if __name__ == "__main__":
    test_queries = [
        "heart anatomy and physiology",
        "guidelines for medical record documentation",
        "public health policies"
    ]

    # Domain to evaluate
    domain = "healthcare_pdfs"

    # Evaluate automatically
    evaluate_retriever(test_queries, domain=domain, top_k=6, threshold=0.55)
