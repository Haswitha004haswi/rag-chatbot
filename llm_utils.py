# llm_utils.py

import openai

# ==============================
# OPENROUTER CONFIG
# ==============================
import os
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

# ==============================
# LLM CLASSIFICATION
# ==============================
def classify_with_llm(query, candidate=None):

    if candidate:
        prompt = f"""
You are a strict medical domain expert.

Is this query STRICTLY about {candidate}?

Return only: {candidate}, Healthcare, Education, or Other.

Query: "{query}"
"""
    else:
        prompt = f"""
Classify the query into:

- Education (medical education: anatomy, physiology, exams)
- Healthcare (disease, patient care, hospitals)
- Other (not medical)

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
        domain = "Other"

    return domain


# ==============================
# LLM ANSWER GENERATION (DB)
# ==============================
def generate_answer(query, context, temperature=0.2, max_tokens=150):
    """
    Generate formatted answer using DB context.
    """

    prompt = f"""
You are a medical assistant.

Answer ONLY using the provided context.

IMPORTANT FORMATTING RULES:
- Write in clear paragraphs.
- Use multiple lines.
- Use bullet points if useful.
- Keep explanation medically accurate.
- Do NOT write in one line.

If answer not found:
"I could not find this in uploaded PDFs."

CONTEXT:
{context}

QUESTION:
{query}
"""

    response = openai.ChatCompletion.create(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response["choices"][0]["message"]["content"].strip()


# ==============================
# LLM WEB SEARCH FALLBACK
# ==============================
def answer_from_web(query, web_text, temperature=0.2, max_tokens=150):
    """
    Generate formatted answer using web context.
    """

    if not web_text.strip():
        return "I could not find this in the web search results."

    prompt = f"""
You are a medical assistant.

Answer ONLY using the provided web search information.

FORMAT RULES:
- Use proper paragraphs.
- Use bullet points if needed.
- Keep answer readable and structured.
- Do NOT write one-line responses.

If answer not found:
"I could not find this in the web search results."

WEB SEARCH TEXT:
{web_text}

QUESTION:
{query}
"""

    response = openai.ChatCompletion.create(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response["choices"][0]["message"]["content"].strip()
