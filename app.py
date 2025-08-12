from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
import io
import re
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY missing. Add it to .env and restart.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Page Config
st.set_page_config(
    page_title="ATS Resume Checker",
    page_icon="📝",
    layout="centered",
)

# ---------- Styling (kept minimal; you can re-add your CSS) ----------
st.markdown("""
<style>
body { background-color: #000; color: #fff; font-family: 'Montserrat', sans-serif; }
.title { text-align:center; font-size:2.4rem; margin-bottom:0.2rem; color:#e6e6e6; }
.subtitle { text-align:center; color:#bdbdbd; margin-top:0; margin-bottom:8px; }
.card { background: rgba(20,20,20,0.85); padding:18px; border-radius:12px; box-shadow: 0 6px 24px rgba(0,0,0,0.6); margin-top:6px; }
.progress-bar { height: 14px; border-radius: 8px; background: linear-gradient(135deg,#4A4A4A,#1A1A1A); }
.small { color:#bdbdbd; font-size:0.9rem; }
.kv { color:#9ef0c1; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">ATS Resume Checker</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste a Job Description or Job Title, upload your resume, and get an ATS score & suggestions</div>', unsafe_allow_html=True)

# ---------- Helper functions ----------
def pdf_to_text_bytes(pdf_bytes_io):
    try:
        reader = PdfReader(pdf_bytes_io)
        text = ""
        for p in reader.pages:
            page_text = p.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception:
        return ""

def extract_text(upload_file):
    if upload_file is None:
        return ""
    file_type = upload_file.type if hasattr(upload_file, "type") else ""
    if "pdf" in file_type:
        with io.BytesIO(upload_file.read()) as b:
            return pdf_to_text_bytes(b)
    elif "text" in file_type or upload_file.name.endswith(".txt"):
        try:
            return upload_file.read().decode("utf-8")
        except Exception:
            return ""
    else:
        # fallback: try to read raw bytes as text
        try:
            return upload_file.read().decode("utf-8")
        except Exception:
            return ""

def simple_tokenize(text):
    tokens = re.findall(r"[A-Za-z0-9\+\#\.\-]+", text.lower())
    # remove numeric-only tokens
    tokens = [t for t in tokens if not re.fullmatch(r"\d+", t)]
    return tokens

def extract_keywords_from_jd(jd_text, top_k=30):
    # Use TF-IDF on sentence-level to find important terms
    docs = [jd_text]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_k)
    try:
        _ = vectorizer.fit_transform(docs)
        return vectorizer.get_feature_names_out().tolist()
    except Exception:
        # fallback: simple frequency
        tokens = simple_tokenize(jd_text)
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_tokens[:top_k]]

def compute_keyword_match(resume_text, jd_keywords):
    resume_tokens = set(simple_tokenize(resume_text))
    if not jd_keywords:
        return 0.0, []
    matched = [kw for kw in jd_keywords if kw.lower() in resume_tokens]
    score = len(matched) / len(jd_keywords)
    return float(score), matched

def compute_tfidf_cosine(resume_text, jd_text):
    try:
        vec = TfidfVectorizer(stop_words='english')
        X = vec.fit_transform([resume_text, jd_text])
        sim = cosine_similarity(X[0:1], X[1:2])[0][0]
        return float(sim)
    except Exception:
        return 0.0

def detect_sections(resume_text):
    # Check for common section headings
    headings = ["experience", "education", "skills", "projects", "certifications", "summary", "contact", "objective"]
    found = {}
    lower = resume_text.lower()
    for h in headings:
        found[h] = (h in lower)
    # bullets presence
    bullets = lower.count("\n•") + lower.count("\n- ") + lower.count("\n* ")
    return found, bullets

def format_score_checks(resume_text):
    score = 0
    reasons = []
    # contact info
    email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", resume_text)
    phone = re.search(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}", resume_text)
    if email:
        score += 1
    else:
        reasons.append("Missing email address in contact section.")
    if phone:
        score += 1
    else:
        reasons.append("Missing phone number in contact section.")
    # required sections
    sections, bullets = detect_sections(resume_text)
    required = ["experience", "education", "skills"]
    sec_count = sum(1 for s in required if sections.get(s))
    score += sec_count  # up to 3
    if sec_count < 3:
        missing = [s for s in required if not sections.get(s)]
        reasons.append("Missing sections: " + ", ".join(missing))
    # bullets
    if bullets >= 2:
        score += 1
    else:
        reasons.append("Few or no bullet points found. Use bullets for achievements.")
    # length checks
    word_count = len(simple_tokenize(resume_text))
    if word_count >= 200:
        score += 1
    else:
        reasons.append("Resume too short; include more detail in experience/results.")
    # compute normalized format score out of 8
    max_points = 8.0
    raw = min(score, max_points)
    fmt_score = (raw / max_points)
    return fmt_score, reasons

def calculate_overall_score(keyword_match, cosine_sim, fmt_score):
    # Weights: keyword_match 45%, cosine_sim 30%, format 25%
    w_keyword = 0.45
    w_cosine = 0.30
    w_fmt = 0.25
    score = (keyword_match * w_keyword + cosine_sim * w_cosine + fmt_score * w_fmt) * 100
    # clamp
    return round(max(0, min(100, score)), 2)

def generate_job_description_from_title(job_title):
    prompt = f"""You are an expert recruiter. Given a job title, produce a concise but detailed job description that includes: key responsibilities, required skills & tools, seniority level, and 6-10 primary keywords that a resume should include to match this role. Job title: "{job_title}".
Format:
JOB_DESCRIPTION:
<one paragraph>
KEYWORDS:
keyword1, keyword2, ..."""
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    resp = model.generate_content(prompt)
    # Return text
    return resp.text

def generate_improvement_text(jd, resume_text, missing_keywords, fmt_reasons, score):
    prompt = f"""
You are an expert resume coach and ATS specialist.

Given this job description:
{jd}

And this resume:
{resume_text[:4000]}

The computed ATS score is {score} out of 100.
Missing keywords: {', '.join(missing_keywords) if missing_keywords else 'None'}
Formatting/issues: {', '.join(fmt_reasons) if fmt_reasons else 'None'}

Provide:
1) A concise explanation of why the score is low or high.
2) Specific, actionable suggestions to increase the ATS score (ordered by priority).
3) A checklist of items to add or edit (bulleted).
Keep answers brief, clear, and practical.
"""
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return resp.text

def rewrite_resume_for_ats(resume_text, jd_text):
    # Ask Gemini to rewrite resume to be ATS friendly, aim for 90+
    prompt = f"""
You are an expert resume writer for ATS systems. Rewrite the following resume to be highly ATS-friendly and aligned to this job description.
Target: make it concise, include keywords from JD, use bullet achievements, include Contact line, Summary, Skills, Experience with bullets (quantified where possible), Education, and Certifications. Keep the style plain text, no tables, no images, no fancy formatting. Job description:
{jd_text}

Resume to rewrite:
{resume_text}

Produce the rewritten resume only.
"""
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    resp = model.generate_content(prompt)
    return resp.text

# ---------- UI Inputs ----------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.form("ats_form"):
        job_title = st.text_input("Optional: Job Title (e.g., 'Frontend Developer')", value="")
        job_description = st.text_area("Paste Job Description (optional)", height=200, placeholder="Paste full job description or leave blank")
        upload_file = st.file_uploader("Upload Your Resume (PDF or TXT)", type=["pdf", "txt"])
        rewrite_opt = st.checkbox("Also rewrite my resume to be ATS-friendly", value=False)
        submitted = st.form_submit_button("Analyze Resume")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Main logic ----------
if submitted:
    if not upload_file:
        st.warning("Please upload your resume file (PDF or TXT).")
        st.stop()

    with st.spinner("Extracting resume text..."):
        resume_text = extract_text(upload_file)

    # If no job description but job title provided, generate JD
    if (not job_description or job_description.strip() == "") and job_title.strip() != "":
        st.info("No job description provided — generating a role-specific job description from the job title...")
        try:
            job_description = generate_job_description_from_title(job_title)
        except Exception as e:
            st.error("Failed to generate job description: " + str(e))
            job_description = ""

    if not job_description or job_description.strip() == "":
        # Use a generic JD for scoring if none provided
        generic_jd = ("General professional role: strong communication, measurable achievements, proficiency in relevant skills,"
                      " clear contact info, experience, education, and skills sections. Keywords vary by industry.")
        job_description = generic_jd

    # Compute keywords and matches
    jd_keywords = extract_keywords_from_jd(job_description, top_k=30)
    kw_score, matched_keywords = compute_keyword_match(resume_text, jd_keywords)
    cosine_sim = compute_tfidf_cosine(resume_text, job_description)
    fmt_score, fmt_reasons = format_score_checks(resume_text)
    overall = calculate_overall_score(kw_score, cosine_sim, fmt_score)

    # Display scores & breakdown
    st.markdown("### 🧾 ATS Score")
    st.progress(int(overall))
    st.write(f"**Overall ATS Score:** {overall} / 100")

    st.markdown("### 🔍 Breakdown")
    st.write(f"- Keyword match: **{round(kw_score*100,2)}%** ({len(matched_keywords)}/{len(jd_keywords)})")
    st.write(f"- Semantic similarity (TF-IDF): **{round(cosine_sim*100,2)}%**")
    st.write(f"- Formatting & baseline checks: **{round(fmt_score*100,2)}%**")

    # Best sections and flaws
    st.markdown("### 👍 Strengths (what looked good)")
    strengths = []
    if any(d for d in ["experience","education","skills"] if d in resume_text.lower()):
        strengths.append("Contains one or more key resume sections (Experience/Education/Skills)")
    if len(simple_tokenize(resume_text)) > 300:
        strengths.append("Sufficient length with detailed points")
    if matched_keywords:
        strengths.append(f"Contains relevant keywords: {', '.join(matched_keywords[:10])}")
    if strengths:
        for s in strengths:
            st.write("- " + s)
    else:
        st.write("- Not much detected — resume may be minimal or poorly structured.")

    st.markdown("### 🔻 Flaws & Issues (why score is low)")
    # list missing keywords
    missing = [k for k in jd_keywords if k not in matched_keywords]
    if missing:
        st.write(f"- Missing keywords ({min(10,len(missing))} shown): {', '.join(missing[:10])}")
    for r in fmt_reasons:
        st.write(f"- {r}")

    # Explain why with Gemini to give a human-friendly reason & prioritized fixes
    with st.spinner("Generating prioritized improvements..."):
        try:
            improvements_text = generate_improvement_text(job_description, resume_text, missing, fmt_reasons, overall)
            st.markdown("### 🛠️ Suggested Improvements (priority ordered)")
            st.write(improvements_text)
        except Exception as e:
            st.error("Failed to generate improvement suggestions: " + str(e))

    # Offer rewrite if requested
    if rewrite_opt:
        with st.spinner("Rewriting resume to be ATS-friendly..."):
            try:
                rewritten = rewrite_resume_for_ats(resume_text, job_description)
                st.markdown("### ✍️ ATS-Optimized Resume (suggested rewrite)")
                st.code(rewritten, language="text")
                # Ask user to confirm replacing? (UI level)
            except Exception as e:
                st.error("Failed to rewrite resume: " + str(e))

    # Final tips (small)
    st.markdown("### ✅ Quick tips to improve score (summary)")
    st.write("""
    - Add a clear Contact line (Name • email • phone • location).
    - Include a short 2-3 line Summary with role & top skills.
    - Add a Skills section listing tools/technologies (match JD keywords).
    - Make Experience bullets achievement-oriented (use numbers).
    - Avoid images, tables, and headers/footers that ATS might ignore.
    - Use plain text headings: Experience, Education, Skills, Projects.
    """)

    st.info("Note: This tool estimates ATS compatibility using keyword and semantic heuristics and provides practical improvements. A real ATS may score differently, but the suggestions here are aimed at making resumes broadly ATS-friendly and readable by recruiters.")
