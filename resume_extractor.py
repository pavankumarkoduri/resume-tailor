"""
resume_extractor.py
====================
Extracts structured, verbatim facts from a resume via a single 'extraction'
LLM call. Extraction (unlike tailoring) is a purely retrieval task: the
model is instructed to copy only what is textually present in the source
document, never to infer or invent job titles, dates, contact details, or
education. This lets the app render a properly structured resume (contact
block, dated experience entries, education) even when the source resume
text is unstructured or table-based -- and it guarantees the later
tailoring step can never blend factual fields (title/company/dates) into
rewritten prose, since those fields are passed through verbatim in code.
"""
from __future__ import annotations
from llm_client import LLMClient

EXTRACT_SYSTEM = """You are a precise resume data-extraction engine. You are
given raw text extracted from a resume file. Extract ONLY facts that are
explicitly present in the text -- never infer, guess, complete, or fabricate
anything that is not written. If a field is not present in the text, return
an empty string ("") or an empty list ([]) for it -- do not omit it from
the JSON structure.

Preserve job titles, company names, and employment dates EXACTLY as written
in the source (same wording, same date format). Split each job's
description into individual bullet points, one array item per bullet or
sentence, preserving the original wording verbatim (do not rewrite,
improve, or merge sentences -- that happens in a separate later step).

Respond with a single valid JSON object matching this exact schema:
{
  "contact": {"name": "", "email": "", "phone": "", "linkedin": "", "location": ""},
  "summary_original": "<verbatim original summary/objective paragraph, or empty string>",
  "skills": ["skill exactly as written", ...],
  "experience": [
    {"title": "", "company": "", "dates": "", "bullets": ["original bullet text", ...]}
  ],
  "projects": [
    {"name": "", "bullets": ["original bullet text", ...]}
  ],
  "education": [
    {"degree": "", "institution": "", "dates": ""}
  ],
  "certifications": ["...", ...]
}
List "experience" entries in the same order they appear in the source
(most recent first, as written). Do not include any text outside the JSON
object."""

EXTRACT_USER_TEMPLATE = """### RAW RESUME TEXT
{resume_text}

Extract the structured facts exactly as instructed. Copy wording verbatim
wherever possible; only split free text into the array fields above."""


def _normalize_education(raw) -> list[dict]:
    """Coerce the 'education' field into a list of {degree, institution,
    dates} dicts, defensively handling models that don't follow the
    requested schema exactly (e.g. returning plain strings like
    "B.S. Computer Science, MIT (2020)" instead of a structured object).
    This prevents a downstream AttributeError when app.py calls .get() on
    each entry, regardless of which model produced the extraction.
    """
    if not raw:
        return []
    normalized = []
    for entry in raw:
        if isinstance(entry, dict):
            normalized.append({
                "degree": str(entry.get("degree", "") or ""),
                "institution": str(entry.get("institution", "") or ""),
                "dates": str(entry.get("dates", "") or ""),
            })
        elif isinstance(entry, str) and entry.strip():
            # Model returned a plain string -- keep the text without losing
            # it, rather than dropping the entry or crashing on .get().
            normalized.append({"degree": entry.strip(), "institution": "", "dates": ""})
    return normalized


def extract_resume_structure(client: LLMClient, resume_text: str) -> dict:
    user_prompt = EXTRACT_USER_TEMPLATE.format(resume_text=resume_text.strip())
    result = client.chat_json(EXTRACT_SYSTEM, user_prompt, max_tokens=2500)
    result["education"] = _normalize_education(result.get("education"))
    return result
