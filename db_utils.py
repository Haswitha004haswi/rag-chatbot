import json
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIGURATION
# -----------------------------
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get Pinecone key from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PERSISTENCE_FILE = "uploaded_chunks.json"

# Domain-specific vector DBs
INDEX_NAMES = {
    "education_pdfs": "medical-education-db",
    "healthcare_pdfs": "healthcare-db"
}

# -----------------------------
# INITIALIZE PINECONE
# -----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
indexes = {domain: pc.Index(name) for domain, name in INDEX_NAMES.items()}

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# -----------------------------
# EMBEDDING FUNCTION
# -----------------------------
def embed_text(text: str):
    """
    Convert text into normalized embedding vector
    """
    return model.encode(text, normalize_embeddings=True).tolist()


# -----------------------------
# SEARCH VECTOR DB (UPDATED)
# -----------------------------
def search_vector_db(
    query: str,
    domain: str,
    top_k: int = 10,
    return_scores: bool = False
):
    """
    Search the relevant domain vector DB.

    Parameters:
        query (str): User question
        domain (str): Vector DB domain name
        top_k (int): Number of chunks to retrieve
        return_scores (bool): If True, also return similarity scores

    Returns:
        - chunks (list[str])
        OR
        - (chunks, scores) if return_scores=True
    """

    if domain not in indexes:
        raise ValueError(
            f"Unknown domain '{domain}'. Choose from {list(indexes.keys())}"
        )

    index = indexes[domain]
    query_embedding = embed_text(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    chunks = []
    scores = []

    for match in results.matches:
        if match.metadata and "text" in match.metadata:
            chunks.append(match.metadata["text"])
            scores.append(match.score)

    if return_scores:
        return chunks, scores

    return chunks


# -----------------------------
# UPLOAD CHUNKS TO DOMAIN
# -----------------------------
def upload_chunks_to_domain(chunks: list, domain: str, start_id: int = 0):
    """
    Upload text chunks into the selected domain vector DB
    """

    if domain not in indexes:
        raise ValueError(
            f"Unknown domain '{domain}'. Choose from {list(indexes.keys())}"
        )

    index = indexes[domain]

    for i, chunk in enumerate(chunks):
        emb = embed_text(chunk)

        index.upsert([
            (
                f"id_{start_id + i}",
                emb,
                {"text": chunk}
            )
        ])


# -----------------------------
# COUNT PDFs PER DOMAIN
# -----------------------------
def count_pdfs_per_domain():
    """
    Count PDFs per domain from persistence file
    """

    try:
        with open(PERSISTENCE_FILE, "r") as f:
            uploaded = json.load(f)
    except FileNotFoundError:
        print("⚠️ No persistence file found. No PDFs uploaded yet.")
        return

    counts = {}

    for pdf_path, domain in uploaded.items():
        counts[domain] = counts.get(domain, 0) + 1

    print("\n📊 PDFs per Vector DB:")
    for domain, count in counts.items():
        print(f"- {domain}: {count} PDFs")


# -----------------------------
# SIMPLE TEST
# -----------------------------
if __name__ == "__main__":
    print("PDF count per domain:")
    count_pdfs_per_domain()
6379