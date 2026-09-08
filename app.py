"""
app.py
======
Resume Tailor — local/hosted Streamlit app.

Pipeline (per JD match):
  1. Extract verbatim structured facts from the resume (contact, titles,
     companies, dates, bullets, education) -- resume_extractor.py
  2. Analyze match against the JD (score, matched/missing skills) -- matcher.py
  3. Tailor ONLY the summary + bullets against the JD -- matcher.py
  4. Merge tailored bullets/summary with the untouched verbatim facts from
     step 1, in pure Python -- resume_builder.py
  5. Export the final structured resume as DOCX/PDF -- doc_export.py

This ensures titles/companies/dates/contact/education can never be altered,
dropped, or blended into bullet prose by the model, and that internal gap
analysis notes never appear in the exported file.

Run:
    streamlit run app.py
"""
import streamlit as st

from resume_parser import extract_text
from llm_client import LLMClient, LLMConfig, resolve_config_from_secrets, is_unreachable_local_config, list_available_models
from matcher import analyze_match, tailor_resume
from resume_extractor import extract_resume_structure
from resume_builder import merge_tailored_resume
from doc_export import build_tailored_docx, docx_bytes_to_pdf, libreoffice_available

st.set_page_config(page_title="Resume Tailor", page_icon="📄", layout="wide")

# ── Session state ────────────────────────────────────────────────────────
for key in ("jd_text", "resume_text", "structured", "analysis", "tailored",
            "final_resume", "tailored_docx", "tailored_pdf"):
    st.session_state.setdefault(key, None)

# ── LLM configuration ────────────────────────────────────────────────────
# On Streamlit Community Cloud, secrets (Settings -> Secrets) provide the
# real API key so end users never see or enter one. Locally (no secrets
# configured), the sidebar lets you point at Ollama/LM Studio/OpenAI freely.
_default_cfg = resolve_config_from_secrets(LLMConfig())
_using_hosted_secret = _default_cfg.api_key not in ("", "ollama") and _default_cfg.base_url != "http://localhost:11434/v1"

def _model_picker(label: str, base_url: str, api_key: str, default_model: str) -> str:
    """Sidebar model selector backed by the provider's live /models list.

    Falls back to a plain text input if the list can't be fetched (bad key,
    unreachable endpoint, or a local backend without a /models route) so the
    app never blocks on this call -- it's a convenience, not a hard
    dependency.
    """
    # When the saved model isn't valid, prefer a well-known, general-purpose
    # chat model over whatever sorts first alphabetically -- provider model
    # catalogs mix in specialty/regional/small models (e.g. Groq's
    # "allam-2-7b", an Arabic-focused 7B model) that technically respond but
    # are unreliable at strict structured JSON output for tasks like this.
    _PREFERRED_SUBSTRINGS = ["llama-3.3-70b", "llama-3.1-70b", "llama-3.1-8b",
                              "mixtral", "gpt-4o", "gemma2-9b"]

    def _pick_best(models: list[str]) -> str:
        for pref in _PREFERRED_SUBSTRINGS:
            for m in models:
                if pref in m.lower():
                    return m
        return models[0]

    available = list_available_models(base_url, api_key)
    if available:
        if default_model in available:
            options = available
            idx = options.index(default_model)
        else:
            # Saved model isn't valid for this key -- show it at the top so
            # the user can see what's configured, but default the actual
            # selection to a real, reliable model instead of the broken one.
            best = _pick_best(available)
            options = [default_model] + available
            idx = options.index(best)
            st.sidebar.caption(
                f"⚠️ `{default_model}` isn't in this key's available model list — "
                f"defaulted to `{best}` below."
            )
        return st.sidebar.selectbox(label, options, index=idx)
    return st.sidebar.text_input(label, value=default_model)


st.sidebar.header("⚙️ LLM Settings")
if _using_hosted_secret:
    st.sidebar.success("Using the server-configured LLM (no key needed).")
    base_url = _default_cfg.base_url
    api_key = _default_cfg.api_key
    model = _model_picker("Model", base_url, api_key, _default_cfg.model)
else:
    st.sidebar.caption(
        "Works with OpenAI, Azure OpenAI, or a fully local model via Ollama/LM Studio."
    )
    base_url = st.sidebar.text_input("Base URL", value=_default_cfg.base_url)
    api_key = st.sidebar.text_input("API Key", value=_default_cfg.api_key, type="password",
                                     help="For local Ollama/LM Studio, any placeholder value works.")
    model = _model_picker("Model", base_url, api_key, _default_cfg.model)

    with st.sidebar.expander("ℹ️ Local model examples"):
        st.code("Ollama:     http://localhost:11434/v1  (api_key: 'ollama')\n"
                "LM Studio:  http://localhost:1234/v1", language="text")

# Guard against the #1 hosted-deploy failure mode: Secrets not set (or the
# key names don't match), so the config silently fell back to the local
# Ollama default -- which no hosted server can ever reach. Surface a clear,
# actionable error up front instead of a raw APIConnectionError deep in an
# LLM call.
_config_unreachable = is_unreachable_local_config(LLMConfig(base_url=base_url, api_key=api_key, model=model))
if _config_unreachable and not _using_hosted_secret:
    st.error(
        "⚠️ **LLM endpoint is set to a local address (`" + base_url + "`).** "
        "If this app is deployed on Streamlit Community Cloud, that address is "
        "never reachable from the server — go to **Settings → Secrets** on "
        "share.streamlit.io and add:\n\n"
        "```toml\nLLM_API_KEY = \"your_groq_or_openai_key\"\n"
        "LLM_BASE_URL = \"https://api.groq.com/openai/v1\"\n"
        "LLM_MODEL = \"llama-3.3-70b-versatile\"\n```\n\n"
        "Then reboot the app. (If you're running this locally with Ollama, "
        "this warning is expected and safe to ignore.)"
    )

if not libreoffice_available():
    st.sidebar.warning(
        "LibreOffice not found — PDF export will be unavailable "
        "(DOCX export always works). Install LibreOffice to enable PDF output."
    )

st.title("📄 Resume Tailor — JD Match & Rewrite")
st.caption("Local web app. Your files are only sent to the LLM endpoint you configure above.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1️⃣ Job Description")
    jd_file = st.file_uploader("Upload JD (PDF or DOCX)", type=["pdf", "docx"], key="jd_upload")
    jd_paste = st.text_area("...or paste JD text", height=180, key="jd_paste")
with col2:
    st.subheader("2️⃣ Your Resume")
    resume_file = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx"], key="resume_upload")


def _resolve_jd_text() -> str:
    if jd_file is not None:
        return extract_text(jd_file.read(), jd_file.name)
    return jd_paste.strip()


def _make_client() -> LLMClient:
    return LLMClient(LLMConfig(base_url=base_url, api_key=api_key, model=model))


def _show_llm_error(e: Exception) -> None:
    """Show a clear, actionable message for the two most common hosted-deploy
    failure modes (bad key / unreachable endpoint) instead of a raw traceback.
    """
    try:
        from openai import AuthenticationError, APIConnectionError, NotFoundError
    except Exception:
        AuthenticationError = APIConnectionError = NotFoundError = ()  # type: ignore

    if isinstance(e, NotFoundError):
        st.error(
            "🚫 **The model `" + model + "` was not found at `" + base_url + "`.**\n\n"
            "This almost always means the API key and the model belong to "
            "different providers (e.g. a Groq key with an OpenAI model name, "
            "or vice versa). Go to **Settings → Secrets** on share.streamlit.io "
            "and make sure `LLM_BASE_URL` and `LLM_MODEL` both match the same "
            "provider as `LLM_API_KEY`:\n\n"
            "```toml\n# Groq\nLLM_API_KEY = \"gsk_...\"\n"
            "LLM_BASE_URL = \"https://api.groq.com/openai/v1\"\n"
            "LLM_MODEL = \"llama-3.3-70b-versatile\"\n\n"
            "# OpenAI\nLLM_API_KEY = \"sk-...\"\n"
            "LLM_BASE_URL = \"https://api.openai.com/v1\"\n"
            "LLM_MODEL = \"gpt-4o-mini\"\n```\n\n"
            "Then click **Reboot app**."
        )
    elif isinstance(e, AuthenticationError):
        st.error(
            "🔑 **The LLM provider rejected the API key** (401 Invalid API Key).\n\n"
            "Go to **Settings → Secrets** on share.streamlit.io and double-check:\n"
            "- The key was copied in full, with no extra spaces or duplicated quotes\n"
            "- It's still active on the provider's dashboard (e.g. console.groq.com/keys) "
            "— generate a fresh one if unsure\n"
            "- `LLM_BASE_URL` matches the provider the key belongs to "
            "(a Groq key will not work against `api.openai.com`, and vice versa)\n\n"
            "Then click **Reboot app** (Secrets changes need a reboot to take effect)."
        )
    elif isinstance(e, APIConnectionError):
        st.error(
            "🔌 **Could not reach the LLM endpoint at `" + base_url + "`.**\n\n"
            "If this app is deployed on Streamlit Community Cloud, a "
            "`localhost`/local address is never reachable from the server — "
            "set `LLM_BASE_URL` in **Settings → Secrets** to a real hosted "
            "endpoint (e.g. `https://api.groq.com/openai/v1`), then reboot."
        )
    else:
        st.exception(e)


analyze_clicked = st.button("🔍 Analyze Match", type="primary", use_container_width=False)

if analyze_clicked:
    try:
        jd_text = _resolve_jd_text()
        if not jd_text:
            st.error("Please upload or paste a job description.")
            st.stop()
        if resume_file is None:
            st.error("Please upload a resume (PDF or DOCX).")
            st.stop()
        resume_text = extract_text(resume_file.read(), resume_file.name)

        st.session_state["jd_text"] = jd_text
        st.session_state["resume_text"] = resume_text

        client = _make_client()
        with st.spinner("Extracting resume structure (contact, jobs, dates, education)..."):
            structured = extract_resume_structure(client, resume_text)
        st.session_state["structured"] = structured

        with st.spinner("Calling LLM for gap analysis..."):
            analysis = analyze_match(client, jd_text, resume_text)
        st.session_state["analysis"] = analysis

        # Reset downstream state from any previous run.
        st.session_state["tailored"] = None
        st.session_state["final_resume"] = None
        st.session_state["tailored_docx"] = None
        st.session_state["tailored_pdf"] = None
    except Exception as e:
        _show_llm_error(e)

# ── Analysis results ─────────────────────────────────────────────────────
analysis = st.session_state.get("analysis")
structured = st.session_state.get("structured")

if analysis:
    st.divider()
    st.subheader("📊 Match Analysis")

    score = analysis.get("match_score", 0)
    st.metric("Match Score", f"{score}/100")
    st.progress(min(max(score, 0), 100) / 100)
    st.write(analysis.get("summary", ""))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**✅ Matched Skills**")
        for s in analysis.get("matched_skills", []):
            st.markdown(f"- {s}")
    with c2:
        st.markdown("**⚠️ Partial / Weak Evidence**")
        for s in analysis.get("partial_skills", []):
            st.markdown(f"- {s}")
    with c3:
        st.markdown("**❌ Missing Skills**")
        for s in analysis.get("missing_skills", []):
            st.markdown(f"- {s}")

    if analysis.get("keyword_gaps"):
        st.markdown("**🔑 Missing Keywords/Phrases**")
        st.write(", ".join(analysis["keyword_gaps"]))

    if analysis.get("recommendations"):
        st.markdown("**💡 Recommendations**")
        for r in analysis["recommendations"]:
            st.markdown(f"- {r}")

if structured:
    st.divider()
    st.subheader("3️⃣ Verify Contact Info & Education")
    st.caption(
        "Extracted automatically from your resume. The tool never invents contact "
        "details — please fill in anything missing before generating the final file."
    )

    contact = structured.get("contact", {}) or {}
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        candidate_name = st.text_input("Full name", value=contact.get("name", ""))
        candidate_email = st.text_input("Email", value=contact.get("email", ""))
    with cc2:
        candidate_phone = st.text_input("Phone", value=contact.get("phone", ""))
        candidate_linkedin = st.text_input("LinkedIn / portfolio URL", value=contact.get("linkedin", ""))
    with cc3:
        candidate_location = st.text_input("Location (city, region)", value=contact.get("location", ""))

    education = structured.get("education") or []
    if education:
        with st.expander("🎓 Education (extracted verbatim)"):
            for edu in education:
                st.write(f"- {edu.get('degree', '')} — {edu.get('institution', '')} ({edu.get('dates', '')})")
    else:
        st.info("No education section detected in the source resume. It will be omitted from the export unless you add your source resume's education section to the text and re-analyze.")

    st.divider()
    st.subheader("4️⃣ Generate Tailored Resume")
    st.caption(
        "Rewrites only your professional summary and experience/project bullets to "
        "align with the JD. Job titles, companies, dates, contact info, and education "
        "are carried over exactly as extracted — never altered or fabricated."
    )

    if st.button("✍️ Tailor My Resume"):
        try:
            client = _make_client()
            with st.spinner("Calling LLM to tailor summary and bullets..."):
                tailored = tailor_resume(client, st.session_state["jd_text"], structured, analysis)
            st.session_state["tailored"] = tailored

            final_resume = merge_tailored_resume(structured, tailored)
            # Overlay the user-verified contact fields (never LLM-authored).
            final_resume["contact"] = {
                "name": candidate_name, "email": candidate_email,
                "phone": candidate_phone, "linkedin": candidate_linkedin,
                "location": candidate_location,
            }
            st.session_state["final_resume"] = final_resume

            docx_bytes = build_tailored_docx(final_resume, candidate_name)
            st.session_state["tailored_docx"] = docx_bytes
            st.session_state["tailored_pdf"] = docx_bytes_to_pdf(docx_bytes)
        except Exception as e:
            _show_llm_error(e)

final_resume = st.session_state.get("final_resume")
if final_resume:
    st.divider()
    st.subheader("📝 Final Resume Preview")

    contact = final_resume.get("contact", {}) or {}
    contact_line = " | ".join(v for v in contact.values() if v and v.strip())
    if contact_line:
        st.caption(contact_line)

    st.markdown("**Professional Summary**")
    st.info(final_resume.get("summary", ""))

    if final_resume.get("skills"):
        st.markdown("**Skills**")
        st.write(" • ".join(final_resume["skills"]))

    if final_resume.get("experience"):
        st.markdown("**Experience**")
        for entry in final_resume["experience"]:
            header = " — ".join(p for p in (entry.get("title"), entry.get("company")) if p)
            st.markdown(f"**{header}**  \n*{entry.get('dates', '')}*")
            for b in entry.get("bullets", []):
                st.markdown(f"- {b}")

    if final_resume.get("projects"):
        st.markdown("**Projects**")
        for proj in final_resume["projects"]:
            st.markdown(f"**{proj.get('name', '')}**")
            for b in proj.get("bullets", []):
                st.markdown(f"- {b}")

    if final_resume.get("education"):
        st.markdown("**Education**")
        for edu in final_resume["education"]:
            line = " — ".join(p for p in (edu.get("degree"), edu.get("institution")) if p)
            st.markdown(f"- {line} ({edu.get('dates', '')})")

    if final_resume.get("notes"):
        st.markdown("**📌 Next Steps to Strengthen This Application** (private — for your eyes only, not included in the exported file)")
        for n in final_resume["notes"]:
            st.markdown(f"- {n}")

    st.divider()
    st.subheader("⬇️ Export")
    dcol, pcol = st.columns(2)
    with dcol:
        if st.session_state.get("tailored_docx"):
            st.download_button(
                "Download DOCX",
                data=st.session_state["tailored_docx"],
                file_name="tailored_resume.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
    with pcol:
        if st.session_state.get("tailored_pdf"):
            st.download_button(
                "Download PDF",
                data=st.session_state["tailored_pdf"],
                file_name="tailored_resume.pdf",
                mime="application/pdf",
            )
        else:
            st.caption("PDF export unavailable (LibreOffice not detected on this machine).")
