🏥 Domain-Aware Medical Chatbot using RAG and Web Search

An intelligent, domain-aware chatbot for Medical Education and Healthcare that delivers accurate, grounded responses using Retrieval-Augmented Generation (RAG), semantic vector search, and real-time web fallback.


📌 Overview
Traditional AI chatbots in healthcare are prone to hallucination and rely on static training data. This project addresses those limitations by building a production-ready RAG pipeline that:

Retrieves verified answers from a Pinecone vector database built from medical PDFs
Falls back to real-time web search (Tavily API) when local knowledge is insufficient
Classifies and rejects out-of-domain queries gracefully
Maintains multi-turn chat memory for contextual conversations


🚀 Features

✅ Domain Classification — Accepts only Medical Education and Healthcare queries; politely rejects everything else
✅ RAG Pipeline — PDF ingestion → chunking → sentence-transformer embeddings → Pinecone vector DB → cosine retrieval → LLM generation
✅ Cross-Encoder Reranking — ms-marco-MiniLM-L-6-v2 reranks retrieved chunks for improved accuracy
✅ Web Search Fallback — Tavily API triggered when retrieval score falls below threshold (0.4)
✅ Chat Memory — Rolling 10-entry conversation history per domain for context-aware responses
✅ FastAPI Backend — Clean REST API with Pydantic validation, auto OpenAPI docs
✅ Streamlit Frontend — Interactive chatbot UI with domain badges and source labels
✅ Cloud Deployment — Deployed on Render (backend) and Streamlit Cloud (frontend)


🏗️ System Architecture
User Query (Streamlit UI)
        │
        ▼
Domain Classifier ──── Out-of-domain? ──► Polite Rejection
        │
        ▼ (Medical / Healthcare)
Pinecone Vector DB (RAG Retrieval)
        │
        ├── Score > 0.4 ──► Cross-Encoder Reranker ──► LLM Generation ──► Response
        │
        └── Score ≤ 0.4 ──► Tavily Web Search ──────► LLM Generation ──► Response

🗂️ Project Structure
domain-aware-medical-chatbot/
│
├── main.py                  # FastAPI app — orchestrates full RAG pipeline
├── domain_classifier.py     # LLM + embedding-based domain classification
├── db_utils.py              # Pinecone vector DB — embed, store, search
├── llm_utils.py             # LLM response generation via OpenRouter
├── reranker.py              # Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
├── memory_rag.py            # Multi-turn memory + query rewriting
├── rasa_greetings.py        # Rule-based greeting and farewell handling
├── retriever_evaluator.py   # Precision@k, Recall@k, MRR evaluation
├── pdf_loader.py            # Batch PDF processing and Pinecone upload
├── streamlit_app.py         # Streamlit chatbot UI
├── data/
│   ├── education_pdfs/      # Medical education PDFs
│   └── healthcare_pdfs/     # Healthcare PDFs
├── uploaded_chunks.json     # Upload persistence tracker
├── requirements.txt
└── .env                     # API keys (not committed)

⚙️ Tech Stack
LayerTechnologyLanguagePython 3.8+BackendFastAPI + UvicornFrontendStreamlitEmbeddingsSentence Transformers (all-MiniLM-L6-v2)Vector DBPineconeRerankerCrossEncoder (ms-marco-MiniLM-L-6-v2)LLMOpenRouter / GPT-3.5-turboWeb SearchTavily APIValidationPydanticDeploymentRender + Streamlit Cloud

📡 API Reference
POST /ask
Submit a medical query.
Request:
json{
  "query": "What are the symptoms of diabetes?"
}
Response:
json{
  "answer": "Diabetes symptoms include increased thirst, frequent urination...",
  "source": "Healthcare PDFs",
  "domain": "Healthcare"
}
Source values: Healthcare PDFs · Education PDFs · Web Search · Greeting · No Source
GET /
Health check — returns server status.

📊 Evaluation Metrics
Run the retriever evaluator:
bashpython retriever_evaluator.py
MetricScore Precision@k 0.33 Recall@k 0.33 MRR 0.67 Domain Classification Accuracy 95%+

🧪 Testing
The system was tested across five levels:

Unit Testing — chunking, embeddings, domain classification, threshold logic
Functional Testing — PDF indexing, query classification, web fallback, UI
Integration Testing — end-to-end pipeline from PDF upload to response display
System Testing — complete system under realistic conditions
API Endpoint Testing — valid queries, out-of-domain rejection, web fallback, error handling


🔮 Future Improvements

Integrate BioBERT embeddings for richer biomedical semantic understanding
Add persistent session management across browser sessions
Implement voice query support
Expand knowledge base with more medical textbooks and clinical guidelines
Add a user feedback mechanism to improve retrieval quality over time
