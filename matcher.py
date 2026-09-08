"""
matcher.py
==========
LLM-driven job-description <-> resume analysis and rewriting.

Two LLM calls (both operate on the structured resume produced by
resume_extractor.py, not raw text -- see app.py for the pipeline):

  1. analyze_match()   -> match score, matched/missing skills, gap summary
  2. tailor_resume()   -> rewritten SUMMARY + BULLETS ONLY. Job titles,
                           companies, dates, education, and contact info are
                           never sent through this step -- they are merged
                           back in code from the extraction step, so they
                           can never be altered, dropped, or blended into
                           prose.
"""
from __future__ import annotations
import json
from llm_client import LLMClient

# ── Prompt: analysis ────────────────────────────────────────────────────
ANALYZE_SYSTEM = """You are an expert technical recruiter and resume analyst.
You compare a candidate's resume against a job description (JD) and produce a
structured, honest gap analysis. You NEVER invent skills or experience that
are not present in the resume text. You only report what is textually
supported (allowing for reasonable synonyms, e.g. "React" == "React.js").

Always respond with a single valid JSON object matching this exact schema:
{
  "match_score": <integer 0-100>,
  "summary": "<2-3 sentence plain-English summary of fit>",
  "matched_skills": ["skill", ...],
  "missing_skills": ["skill", ...],
  "partial_skills": ["skill present but weakly evidenced", ...],
  "keyword_gaps": ["important JD keyword/phrase absent from resume", ...],
  "recommendations": ["specific, actionable suggestion", ...]
}
Do not include any text outside the JSON object."""

ANALYZE_USER_TEMPLATE = """### JOB DESCRIPTION
{jd_text}

### CANDIDATE RESUME
{resume_text}

Analyze fit between the resume and the job description. Extract skills and
requirements from the JD (technical skills, tools, years of experience,
certifications, soft skills) and check each against the resume. Return the
JSON object as instructed."""


# ── Prompt: tailoring / rewriting (structured, bullets + summary ONLY) ──
TAILOR_SYSTEM = """You are an expert resume writer. You are given a
candidate's resume already split into structured fields (job titles,
companies, dates, and bullet points extracted verbatim from their real
resume) plus a target job description (JD). Your ONLY job is to rewrite:
  (a) a tailored professional summary, and
  (b) the bullet points for each experience entry and project (returned in
      the SAME order, matched by the "id" given for each entry).

PRIMARY GOAL -- align to the job description. This is the entire point of
this rewrite, and it takes priority over everything else below:
  - Read the JD closely and identify its key requirements: required skills,
    tools/technologies, years of experience, responsibilities, domain
    knowledge, and soft skills.
  - For every bullet, actively ask "does the original bullet for this entry
    support restating this in the JD's own terminology?" If yes, rewrite it
    using that terminology (e.g. if the JD says "stakeholder management"
    and the original bullet describes coordinating with clients/teams, use
    "stakeholder management"). This is not optional polish -- it is the
    main task.
  - Within each entry, reorder bullets so the ones most relevant to the JD's
    stated requirements come first, and less relevant ones come later.
  - In the summary, foreground the candidate's experience/skills that most
    directly match the JD's top requirements; do not lead with generic
    background unrelated to this specific role.
  - Naturally work in exact keywords/phrases from the JD wherever the
    resume already provides real, truthful evidence for them (this matters
    for both human recruiters and ATS keyword screening) -- but never at
    the cost of fabricating something the resume doesn't support.

You may NOT invent new employers, job titles, dates, degrees,
certifications, or bullets describing work not implied by the original
bullets for that entry. You may NOT claim skills/tools the original bullets
for that entry give no evidence of. You may NOT fabricate metrics that are
not present in the original bullets. Do not mention the job description,
the candidate's fit, or any gap analysis inside the summary or bullets --
those belong only in the separate "notes" field.

HOW TO WRITE IT -- once the JD-aligned content/emphasis above is decided,
write it like a specific human describing their own work, not generic AI
copy. This resume must be indistinguishable from one written directly by
the candidate, and must read naturally to recruiters in ANY
country/English variant (not just US English):
  - Vary sentence/bullet structure and length; do not start every bullet
    with an identical "Power Verb + metric" template. Mix short, direct
    bullets with occasional longer ones, the way a real person writes.
  - Avoid overused AI-sounding words/phrases entirely: "leverage",
    "leveraged", "utilize", "utilized", "spearheaded", "seamless",
    "seamlessly", "robust", "synergy", "cutting-edge", "dynamic",
    "fast-paced environment", "passionate about", "proven track record",
    "results-driven", "game-changer", "delve", "furthermore", "moreover",
    "in today's [X] landscape", "unlock", "elevate", "empower", "holistic",
    "streamline" (unless the original bullet already used it), "harness",
    "bespoke", "tailored solutions", "paradigm", "ecosystem" (unless
    referring to a literal technical ecosystem). Use plain, specific,
    everyday language instead.
  - Do not use em-dashes as a stylistic tic; use plain punctuation (commas,
    periods, "and"). Avoid triplet lists ("X, Y, and Z" used repeatedly as a
    crutch) and avoid ending multiple bullets with the exact same sentence
    shape (e.g. every bullet ending in "resulting in X% improvement").
  - Write in a neutral, international English register: do not assume
    US-only idioms, spelling, or cultural references. Keep whatever
    spelling convention (US/UK/AU/etc.) the original resume already used --
    do not silently convert between them.
  - Each bullet must be ONE self-contained sentence about ONE accomplishment
    or duty. Never combine job title, company, dates, or gap commentary
    into a bullet -- those are handled separately and must not appear in
    bullet text at all.
  - Prefer concrete, specific detail (what was built, for whom, at what
    scale) over vague claims of impact. A slightly imperfect, specific
    sentence reads more human than a polished generic one.

Always respond with a single valid JSON object matching this exact schema:
{
  "tailored_summary": "<3-4 sentence professional summary tailored to the JD, based only on the candidate's real background>",
  "experience": [
    {"id": <same id as input>, "bullets": ["rewritten bullet 1", "rewritten bullet 2", "..."]}
  ],
  "projects": [
    {"id": <same id as input>, "bullets": ["rewritten bullet 1", "..."]}
  ],
  "skills_section": ["<curated/reordered skill, formatted exactly as input>", "..."],
  "notes": ["<honest note about a JD requirement that could not be addressed given the candidate's real experience>", "..."]
}
Do not include any text outside the JSON object."""

TAILOR_USER_TEMPLATE = """### JOB DESCRIPTION (the target role -- align the summary and every bullet to this)
{jd_text}

### CANDIDATE'S STRUCTURED EXPERIENCE (rewrite bullets only; titles/companies/dates shown for context only and must not appear inside bullet text)
{structured_experience_json}

### CANDIDATE'S SKILLS (reorder/curate only; do not add skills not listed)
{skills_json}

### KNOWN GAPS (from prior analysis -- do not fabricate fixes; only mention honestly in "notes")
Missing skills: {missing_skills}
Keyword gaps: {keyword_gaps}

Rewrite the summary and bullets so they are clearly and specifically aligned
to the job description above -- reusing its terminology wherever the
candidate's real experience supports it, and prioritizing/reordering
content by relevance to it. Follow the system instructions for exactly how
to do this truthfully and how to write it naturally. Return the JSON object
as instructed, using the same "id" values given above for each
experience/project entry."""


def analyze_match(client: LLMClient, jd_text: str, resume_text: str) -> dict:
    user_prompt = ANALYZE_USER_TEMPLATE.format(jd_text=str(jd_text).strip(), resume_text=str(resume_text).strip())
    return client.chat_json(ANALYZE_SYSTEM, user_prompt, max_tokens=1800)


def tailor_resume(client: LLMClient, jd_text: str, structured: dict, analysis: dict) -> dict:
    """Rewrite only the summary + bullets of an already-extracted, structured
    resume (see resume_extractor.py). Titles/companies/dates/education/
    contact info are never passed through the LLM here and are merged back
    in app.py from the extraction step untouched.
    """
    exp_for_prompt = [
        {"id": i, "title": e.get("title", ""), "company": e.get("company", ""),
         "dates": e.get("dates", ""), "bullets": e.get("bullets", [])}
        for i, e in enumerate(structured.get("experience", []))
    ]
    proj_for_prompt = [
        {"id": i, "name": p.get("name", ""), "bullets": p.get("bullets", [])}
        for i, p in enumerate(structured.get("projects", []))
    ]
    combined = {"experience": exp_for_prompt, "projects": proj_for_prompt}

    missing = analysis.get("missing_skills", []) or []
    gaps = analysis.get("keyword_gaps", []) or []

    user_prompt = TAILOR_USER_TEMPLATE.format(
        jd_text=str(jd_text).strip(),
        structured_experience_json=json.dumps(combined, indent=2),
        skills_json=json.dumps(structured.get("skills", []), indent=2),
        missing_skills=", ".join(str(m) for m in missing) or "none",
        keyword_gaps=", ".join(str(g) for g in gaps) or "none",
    )
    return client.chat_json(TAILOR_SYSTEM, user_prompt, max_tokens=3000)
