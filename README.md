# Resume Tailor

A local Streamlit app that:
1. Extracts text from a **Job Description** and **Resume** (PDF or DOCX only).
2. Sends both to an LLM to compute a **match score**, matched/missing skills,
   and keyword gaps.
3. Uses the LLM to **rewrite/tailor** the resume to better align with the JD —
   without fabricating skills, employers, dates, or metrics.
4. Exports the tailored resume as **DOCX** (always) and **PDF** (if
   LibreOffice is installed).

## Setup

```bash
cd resume_tailor
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Optional: enable PDF export
Install LibreOffice (free) so DOCX → PDF conversion works headlessly:
- macOS: `brew install --cask libreoffice`
- Windows: download from https://www.libreoffice.org/download/
- Linux: `sudo apt install libreoffice`

If LibreOffice isn't installed, the app still works — you just won't get a
PDF download button (DOCX export is unaffected).

## Run

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`. Nothing is
sent anywhere except:
- the LLM API call (to whichever endpoint you configure in the sidebar).

## LLM backend options (set in the sidebar, no code changes needed)

| Backend        | Base URL                          | API Key            |
|----------------|------------------------------------|---------------------|
| OpenAI         | `https://api.openai.com/v1`        | your real API key   |
| Azure OpenAI   | your resource endpoint + `/openai` | your Azure key      |
| Ollama (local) | `http://localhost:11434/v1`        | any placeholder     |
| LM Studio      | `http://localhost:1234/v1`         | any placeholder     |

Using Ollama or LM Studio keeps **100% of your data local** — no resume or
JD content ever leaves your machine.

## Files

- `app.py` — Streamlit UI and orchestration
- `resume_parser.py` — PDF/DOCX text extraction
- `llm_client.py` — thin OpenAI-compatible chat client (JSON-mode aware)
- `matcher.py` — prompts + calls for match analysis and resume tailoring
- `doc_export.py` — builds the tailored DOCX and converts it to PDF via
  headless LibreOffice

## Notes / limitations

- Scanned/image-based PDFs are not supported (no OCR) — text-based PDFs only.
- The LLM is instructed not to fabricate experience; always review the
  "Next Steps to Strengthen This Application" section before submitting a
  tailored resume anywhere.
- Match scoring is inherently approximate — treat it as a directional signal,
  not a guarantee of ATS behavior.
