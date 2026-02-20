import os
import json
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import PyPDF2
import pdfplumber
from tqdm import tqdm

# -----------------------------
# CONFIGURATION
# -----------------------------
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get Pinecone key from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PDF_FOLDER = "data"                   # root folder containing domain subfolders
CHUNK_SIZE = 300                      # words per chunk
PAGES_PER_CHUNK = 10                  # pages per chunk for large PDFs
BATCH_SIZE = 50                       # Pinecone upsert batch size
PERSISTENCE_FILE = "uploaded_chunks.json"

# Map folders to Pinecone indexes
INDEX_NAMES = {
    "education_pdfs": "medical-education-db",
    "healthcare_pdfs": "healthcare-db"
}

# -----------------------------
# INITIALIZE PINECONE
# -----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
indexes = {folder: pc.Index(index_name) for folder, index_name in INDEX_NAMES.items()}

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# PERSISTENCE FILE FUNCTIONS
# -----------------------------
def load_persistence():
    if not os.path.exists(PERSISTENCE_FILE):
        return {}
    try:
        with open(PERSISTENCE_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}

def save_persistence(data):
    with open(PERSISTENCE_FILE, "w") as f:
        json.dump(data, f, indent=2)

# -----------------------------
# PDF FUNCTIONS
# -----------------------------
def extract_pdf_in_chunks(pdf_path, pages_per_chunk=PAGES_PER_CHUNK):
    """Extract PDF text in chunks of pages_per_chunk."""
    chunks = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            for i in range(0, num_pages, pages_per_chunk):
                text_chunk = ""
                for page in reader.pages[i:i + pages_per_chunk]:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_chunk += page_text + "\n"
                    except Exception:
                        continue  # skip corrupted pages
                if text_chunk.strip():
                    chunks.append(text_chunk)
    except Exception as e:
        print(f"⚠️ PyPDF2 failed for {pdf_path}: {e}")
        # fallback to pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if text.strip():
                    chunks.extend(split_text_into_chunks(text))
        except Exception as e2:
            print(f"⚠️ pdfplumber also failed for {pdf_path}: {e2}")
    return chunks

def split_text_into_chunks(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_text(text):
    return model.encode(text, normalize_embeddings=True).tolist()

def get_relative_path(path):
    return os.path.relpath(path, PDF_FOLDER).replace("\\", "/")

def classify_pdf_domain(pdf_path):
    parent_folder = os.path.basename(os.path.dirname(pdf_path)).lower()
    if parent_folder in INDEX_NAMES:
        return parent_folder
    return "education_pdfs"

# -----------------------------
# MAIN UPLOAD FUNCTION
# -----------------------------
def process_pdfs_and_upload():
    uploaded_pdfs = load_persistence()
    pdf_files = []

    # Collect all PDFs in subfolders
    for root, _, files in os.walk(PDF_FOLDER):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    total_chunks = 0
    new_pdfs = 0

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        rel_path = get_relative_path(pdf_path)

        if rel_path in uploaded_pdfs:
            print(f"⏩ Skipping already uploaded: {rel_path}")
            continue

        domain = classify_pdf_domain(pdf_path)
        index = indexes[domain]

        print(f"⬆️ Uploading: {rel_path} → {domain}")

        chunks = extract_pdf_in_chunks(pdf_path)
        if not chunks:
            print(f"⚠️ Empty or unreadable PDF skipped: {rel_path}")
            continue

        # Further split large chunks if necessary
        all_chunks = []
        for chunk in chunks:
            all_chunks.extend(split_text_into_chunks(chunk))

        # Batch upsert
        batch = []
        for i, chunk in enumerate(all_chunks):
            emb = embed_text(chunk)
            batch.append((f"{rel_path}_{i}", emb, {"text": chunk}))

            if len(batch) >= BATCH_SIZE:
                index.upsert(batch)
                batch = []

        if batch:
            index.upsert(batch)

        uploaded_pdfs[rel_path] = domain
        new_pdfs += 1
        total_chunks += len(all_chunks)

    save_persistence(uploaded_pdfs)

    print("\n==========================")
    print(f"Total PDFs found: {len(pdf_files)}")
    print(f"New PDFs uploaded: {new_pdfs}")
    print(f"Total chunks uploaded: {total_chunks}")
    print("==========================\n")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    process_pdfs_and_upload()
