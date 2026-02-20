from sentence_transformers import CrossEncoder

# load reranker model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_chunks(query, chunks, top_k=5):
    """
    Rerank retrieved chunks using cross-encoder
    """

    pairs = [[query, chunk] for chunk in chunks]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [chunk for chunk, _ in ranked[:top_k]]
