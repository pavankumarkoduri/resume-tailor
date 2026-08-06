"""
resume_builder.py
==================
Pure-Python (no LLM) merge step: combines the verbatim-extracted resume
structure (contact info, job titles, companies, dates, education --
resume_extractor.py) with the LLM's tailored summary/bullets (matcher.py)
into one final structured resume dict, ready for doc_export.py.

This merge happens in code, not via another LLM call, specifically so that
factual fields (titles, companies, dates, contact info, education) can
NEVER be altered, dropped, or blended into prose by the model -- they pass
through untouched from extraction.
"""
from __future__ import annotations
import re

LEVEL_WORDS = {"basic", "beginner", "intermediate", "advanced", "expert", "proficient"}


def _s(v) -> str:
    """Coerce a value to a plain string; defends against the LLM occasionally
    returning a nested object/dict for a field that should be a string."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return ""
    return str(v)


def merge_tailored_resume(structured: dict, tailored: dict) -> dict:
    """Return a final structured resume dict combining verbatim facts with
    the LLM's tailored summary/bullets."""
    tailored_exp_by_id = {e.get("id"): e.get("bullets", []) for e in tailored.get("experience", [])}
    tailored_proj_by_id = {p.get("id"): p.get("bullets", []) for p in tailored.get("projects", [])}

    final_experience = []
    for i, entry in enumerate(structured.get("experience", [])):
        bullets = tailored_exp_by_id.get(i) or entry.get("bullets", [])
        final_experience.append({
            "title": _s(entry.get("title", "")).strip(),
            "company": _s(entry.get("company", "")).strip(),
            "dates": _s(entry.get("dates", "")).strip(),
            "bullets": [_s(b).strip() for b in bullets if b and _s(b).strip()],
        })

    final_projects = []
    for i, proj in enumerate(structured.get("projects", [])):
        bullets = tailored_proj_by_id.get(i) or proj.get("bullets", [])
        final_projects.append({
            "name": _s(proj.get("name", "")).strip(),
            "bullets": [_s(b).strip() for b in bullets if b and _s(b).strip()],
        })

    skills_raw = tailored.get("skills_section") or structured.get("skills", [])
    skills = [_normalize_skill(_s(s)) for s in skills_raw if s and _s(s).strip()]

    return {
        "contact": structured.get("contact", {}) or {},
        "summary": _s(tailored.get("tailored_summary") or structured.get("summary_original") or "").strip(),
        "skills": skills,
        "experience": final_experience,
        "projects": final_projects,
        "education": structured.get("education", []) or [],
        "certifications": structured.get("certifications", []) or [],
        "notes": tailored.get("notes", []) or [],
    }


def _normalize_skill(skill: str) -> str:
    """Normalize 'Skill(Level)' / 'skill (level)' / etc. into a consistent
    'Skill (Level)' format so output never has the inconsistent casing/
    spacing an LLM can introduce (e.g. 'NFC(Advanced)' vs 'SIEM (intermediate)')."""
    s = skill.strip()
    m = re.match(r"^(.*?)\s*\(\s*([A-Za-z]+)\s*\)\s*$", s)
    if not m:
        return s
    name, level = m.group(1).strip(), m.group(2).strip()
    if level.lower() in LEVEL_WORDS:
        return f"{name} ({level.capitalize()})"
    return s
