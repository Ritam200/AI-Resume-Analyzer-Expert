# app.py
import os
import json
import uuid
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# Optional audio libs (if you install)
# import speech_recognition as sr
# from pydub import AudioSegment

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ------------------- Utilities -------------------
MODEL_EMBED = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

DATA_DIR = "user_data"
os.makedirs(DATA_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                p = page.extract_text()
                if p:
                    text += p + "\n"
        if text.strip():
            return text.strip()
    except Exception as e:
        print("pdfplumber failed:", e)

    # fallback OCR
    try:
        images = convert_from_path(pdf_path)
        for img in images:
            page_text = pytesseract.image_to_string(img)
            text += page_text + "\n"
    except Exception as e:
        print("OCR failed:", e)

    return text.strip()

def embed_texts(texts):
    if isinstance(texts, str):
        texts = [texts]
    emb = MODEL_EMBED.encode(texts, convert_to_numpy=True)
    return emb

def cosine_sim(a, b):
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])

def save_user_progress(user_id, record):
    path = os.path.join(DATA_DIR, f"{user_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {"sessions": []}
    data["sessions"].append(record)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_user_progress(user_id):
    path = os.path.join(DATA_DIR, f"{user_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"sessions": []}

# ------------------- Basic Resume Analyzer using Gemini -------------------
def analyze_resume_with_gemini(resume_text, job_description=None):
    if not resume_text:
        return "No resume text provided."
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
You are an experienced HR and technical interviewer. Review the resume below.
Resume:
{resume_text}

If a job description is provided, compare strengths/weaknesses to the JD.
Job Description:
{job_description if job_description else 'N/A'}

Provide:
1) Short one-line summary
2) Top skills present
3) Missing skills relative to JD (if JD provided)
4) Suggested courses/resources (3)
5) Quick ATS keywords to add
6) Strengths and weaknesses (bulleted)
"""
    res = model.generate_content(prompt)
    return res.text.strip()

# ------------------- Simple Role-specific Question Bank + Vector Retrieval -------------------
# A starter question bank. You should expand these and/or load from files.
SAMPLE_QBANK = [
    {"role": "Software Developer", "question": "Explain the concept of OOP and give examples."},
    {"role": "Software Developer", "question": "What is REST API and how is it different from SOAP?"},
    {"role": "Software Developer", "question": "Write a function to reverse a string in Python."},
    {"role": "Data Scientist", "question": "Explain bias-variance tradeoff."},
    {"role": "Data Scientist", "question": "How do you handle missing values in a dataset?"},
    {"role": "DevOps Engineer", "question": "What is CI/CD and why is it important?"},
    {"role": "HR Manager", "question": "Describe a time you handled a conflict between team members."},
    {"role": "HR Manager", "question": "How would you structure a hiring process for junior engineers?"}
]

# Build simple in-memory embeddings for retrieval
for item in SAMPLE_QBANK:
    item["embed"] = embed_texts(item["question"])[0]

def retrieve_questions(role, k=5):
    # filter by role and then return top-k by simple order or similarity to role string
    candidates = [q for q in SAMPLE_QBANK if q["role"] == role]
    if not candidates:
        # fallback: return some generic questions
        candidates = SAMPLE_QBANK[:k]
    return candidates[:k]

# ------------------- Model Answer Generation (Gemini) -------------------
def generate_model_answer(question, resume_text=None):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
You are a senior interviewer and subject-matter expert. Provide:
1) An ideal model answer for the question.
2) A short rubric describing what the interviewer expects (3 bullet points).
Question: {question}
Candidate Resume (for personalization): {resume_text if resume_text else 'N/A'}
"""
    resp = model.generate_content(prompt)
    return resp.text.strip()

# ------------------- Answer Evaluation -------------------
def evaluate_answer(candidate_answer, model_answer):
    # 1) Embedding similarity score
    emb_c = embed_texts(candidate_answer)[0]
    emb_m = embed_texts(model_answer)[0]
    sim = cosine_similarity(emb_c.reshape(1,-1), emb_m.reshape(1,-1))[0][0]
    score_percent = round(float(sim) * 100, 1)

    # 2) AI-driven feedback (concise) using Gemini (short call)
    model = genai.GenerativeModel("gemini-1.5-flash")
    feedback_prompt = f"""
You are an interviewer. Given the model ideal answer and candidate answer below, give:
- A short score justification (one sentence).
- 3 concise improvement tips for the candidate.
Model Answer:
{model_answer}

Candidate Answer:
{candidate_answer}
"""
    resp = model.generate_content(feedback_prompt)
    feedback_text = resp.text.strip()
    return {"score": score_percent, "feedback": feedback_text}

# ------------------- ATS Check -------------------
def ats_keyword_match(resume_text, job_description):
    if not job_description:
        return {"match_percent": None, "missing_keywords": []}
    # simple keyword extraction: split JD into words, pick nouns/keywords by frequency (simple)
    jd_tokens = [w.strip(".,()[]").lower() for w in job_description.split() if len(w) > 3]
    resume_tokens = [w.strip(".,()[]").lower() for w in resume_text.split() if len(w) > 3]
    jd_unique = set(jd_tokens)
    resume_unique = set(resume_tokens)
    matched = jd_unique.intersection(resume_unique)
    match_percent = round(len(matched) / (len(jd_unique) if len(jd_unique) else 1) * 100, 1)
    missing = list(jd_unique - matched)[:30]
    return {"match_percent": match_percent, "missing_keywords": missing}

# ------------------- Learning Path (simple mapping) -------------------
SKILL_COURSES = {
    "python": ["Coursera: Python for Everybody", "freeCodeCamp Python Tutorial", "Udemy: Complete Python Bootcamp"],
    "machine learning": ["Coursera: Machine Learning by Andrew Ng", "fast.ai Practical Deep Learning"],
    "sql": ["Mode SQL Tutorial", "Khan Academy SQL", "Udemy: SQL Bootcamp"],
    "devops": ["Udemy: Docker & Kubernetes", "Coursera: DevOps Culture and Mindset"],
    "data analysis": ["Google Data Analytics Certificate", "Pandas documentation tutorials"]
}

def suggest_courses(missing_skills):
    suggestions = []
    ms_lower = [s.lower() for s in missing_skills]
    for key, courses in SKILL_COURSES.items():
        if key in ms_lower or any(key in s for s in ms_lower):
            suggestions.extend(courses)
    # add some generic suggestions if none matched
    if not suggestions:
        suggestions = ["Coursera / Udemy top courses in the relevant domain", "YouTube tutorials and documentation"]
    return suggestions[:5]

# ------------------- PDF Report Generation -------------------
def generate_pdf_report(filename, summary_text, qna_results, overall_score):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 40, "Interview Readiness Report")
    c.setFont("Helvetica", 10)
    c.drawString(40, height - 60, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    c.drawString(40, height - 80, f"Overall Score: {overall_score}")
    text = c.beginText(40, height - 110)
    text.setLeading(14)
    for line in summary_text.split("\n"):
        text.textLine(line)
    c.drawText(text)

    y = height - 300
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Q&A Summary & Feedback:")
    y -= 20
    c.setFont("Helvetica", 9)
    for qitem in qna_results:
        if y < 60:
            c.showPage()
            y = height - 40
        c.drawString(40, y, f"Q: {qitem['question']}")
        y -= 14
        c.drawString(60, y, f"Score: {qitem['score']}")
        y -= 14
        for line in qitem['feedback'].split("\n"):
            c.drawString(60, y, line[:100])
            y -= 12
        y -= 8
    c.save()

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="AI Resume + Interview Trainer", layout="wide")
st.title("AI Resume Analyzer & Interview Trainer (Extended)")

tabs = st.tabs(["Resume Analyzer", "Interview Trainer", "Mock Interview (Practice)", "Progress & Report"])

# Shared upload
with tabs[0]:
    st.header("Upload Resume & Analyze")
    col1, col2 = st.columns([2,1])
    with col1:
        uploaded_file = st.file_uploader("Upload your resume (PDF):", type=["pdf"])
        job_description = st.text_area("Optional: Paste Job Description (for JD match & ATS):", height=150)
    with col2:
        st.write("Profile / User ID (for saving progress):")
        user_id = st.text_input("Enter an identifier (email or username):", value="anonymous_user")
        if st.button("Analyze Resume"):
            if not uploaded_file:
                st.warning("Please upload a resume PDF.")
            else:
                with open("uploaded_resume.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                resume_text = extract_text_from_pdf("uploaded_resume.pdf")
                st.subheader("Resume Extract (first 800 chars):")
                st.write(resume_text[:800] + ("..." if len(resume_text) > 800 else ""))

                # Gemini analysis
                with st.spinner("Running AI resume analysis..."):
                    analysis_text = analyze_resume_with_gemini(resume_text, job_description)
                st.subheader("AI Resume Analysis:")
                st.write(analysis_text)

                # ATS
                ats = ats_keyword_match(resume_text, job_description)
                if ats["match_percent"] is not None:
                    st.subheader("ATS Keyword Match")
                    st.write(f"Match: {ats['match_percent']}%")
                    st.write("Top missing keywords (samples):", ats["missing_keywords"][:10])

                # Suggest courses from missing keywords
                suggested = suggest_courses(ats["missing_keywords"] if ats["missing_keywords"] else [])
                st.subheader("Suggested Courses / Resources:")
                for s in suggested:
                    st.write("- " + s)

                # Save minimal analysis to user file
                rec = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "resume_analysis",
                    "ats_match": ats.get("match_percent"),
                    "suggested_courses": suggested
                }
                save_user_progress(user_id, rec)
                st.success("Analysis saved to your progress profile.")

# Interview Trainer tab
with tabs[1]:
    st.header("Interview Trainer (Role-based, RAG style)")
    role = st.selectbox("Select Role", options=list({item["role"] for item in SAMPLE_QBANK}))
    resume_text_input = st.text_area("Paste resume text here (or leave empty to use last uploaded):", height=150)
    if st.button("Generate Role-specific Q&A"):
        resume_text = resume_text_input
        if not resume_text and os.path.exists("uploaded_resume.pdf"):
            resume_text = extract_text_from_pdf("uploaded_resume.pdf")
        if not resume_text:
            st.warning("Please provide resume text or upload a resume in the Resume Analyzer tab.")
        else:
            with st.spinner("Retrieving questions and generating model answers..."):
                qs = retrieve_questions(role, k=5)
                qna_results = []
                for q in qs:
                    question = q["question"]
                    model_ans = generate_model_answer(question, resume_text)
                    # present to user
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**Model answer / rubric:**\n\n{model_ans}")
                    qna_results.append({"question": question, "model_answer": model_ans})
                st.success("Questions & model answers generated.")

# Mock Interview (Practice)
with tabs[2]:
    st.header("Mock Interview (Practice Mode)")
    st.write("You will get questions one-by-one. Type your answer (or upload audio). System will score & give feedback.")
    user_id2 = st.text_input("Profile ID (for saving this session):", value="anonymous_user_practice", key="uid2")
    role2 = st.selectbox("Choose role for practice:", options=list({item["role"] for item in SAMPLE_QBANK}), key="role2")
    k_qs = st.slider("Number of questions:", min_value=1, max_value=7, value=3)
    if st.button("Start Practice Session"):
        resume_text = ""
        if os.path.exists("uploaded_resume.pdf"):
            resume_text = extract_text_from_pdf("uploaded_resume.pdf")
        questions = retrieve_questions(role2, k=k_qs)
        session_qna = []
        for idx, q in enumerate(questions, 1):
            st.markdown(f"### Q{idx}: {q['question']}")
            # Allow text answer
            ans = st.text_area(f"Your answer to Q{idx} (text):", key=f"ans_{idx}", height=120)
            # optional audio upload (not auto-processed unless you enable STT)
            audio_file = st.file_uploader(f"Or upload audio answer for Q{idx} (optional):", type=["wav","mp3"], key=f"audio_{idx}")
            candidate_answer = ans
            # If audio uploaded and speechrecognition installed, convert (optional)
            if audio_file is not None:
                # save temporarily
                tmp_audio_path = f"tmp_{user_id2}_{idx}_{uuid.uuid4().hex}.wav"
                with open(tmp_audio_path, "wb") as f:
                    f.write(audio_file.getbuffer())
                # If speechrecognition is installed, one could run recognition here
                # r = sr.Recognizer()
                # with sr.AudioFile(tmp_audio_path) as source:
                #     audio_data = r.record(source)
                # try:
                #     candidate_answer = r.recognize_google(audio_data)
                # except Exception as e:
                #     st.warning("Audio to text failed; please type your answer.")
                st.info("Audio uploaded (STT not active here). Make sure to type as well if STT disabled.")

            if st.button(f"Submit Answer Q{idx}", key=f"sub_{idx}"):
                with st.spinner("Evaluating your answer..."):
                    model_answer = generate_model_answer(q["question"], resume_text)
                    eval_res = evaluate_answer(candidate_answer if candidate_answer else "No answer provided.", model_answer)
                    st.markdown(f"**Score:** {eval_res['score']} / 100")
                    st.markdown("**Feedback (AI):**")
                    st.write(eval_res["feedback"])
                    session_qna.append({
                        "question": q["question"],
                        "candidate_answer": candidate_answer,
                        "score": eval_res["score"],
                        "feedback": eval_res["feedback"],
                        "model_answer": model_answer
                    })
        # After the loop, summarize & save
        if session_qna:
            avg_score = round(sum(item["score"] for item in session_qna)/len(session_qna), 1)
            st.success(f"Session complete. Average score: {avg_score}")
            rec = {"timestamp": datetime.utcnow().isoformat(), "type": "practice_session",
                   "role": role2, "avg_score": avg_score, "qna": session_qna}
            save_user_progress(user_id2, rec)
            st.info("Session saved to your profile.")

# Progress & Report
with tabs[3]:
    st.header("Progress and Generate Report")
    prof_id = st.text_input("Enter your Profile ID to view progress:", value="anonymous_user")
    if st.button("Load Progress"):
        data = load_user_progress(prof_id)
        st.write(data)
        # show basic analytics
        sessions = data.get("sessions", [])
        practice_scores = [s["avg_score"] for s in sessions if s.get("type")=="practice_session"]
        import matplotlib.pyplot as plt
        if practice_scores:
            fig = plt.figure()
            plt.plot(range(1, len(practice_scores)+1), practice_scores, marker='o')
            plt.title("Practice Session Scores Over Time")
            plt.xlabel("Session #")
            plt.ylabel("Average Score")
            st.pyplot(fig)
        else:
            st.info("No practice session data yet.")

    st.write("---")
    st.write("Generate a PDF Report from last practice session")
    report_id = st.text_input("Profile ID for report:", value="anonymous_user")
    if st.button("Generate PDF Report"):
        data = load_user_progress(report_id)
        sessions = [s for s in data.get("sessions", []) if s.get("type")=="practice_session"]
        if not sessions:
            st.warning("No practice sessions found for this profile.")
        else:
            last = sessions[-1]
            summary = f"Profile: {report_id}\nRole: {last.get('role')}\nAvg Score: {last.get('avg_score')}\nGenerated: {datetime.utcnow().isoformat()}"
            qna = []
            for q in last.get("qna", []):
                qna.append({"question": q["question"], "score": q["score"], "feedback": q["feedback"]})
            fname = f"report_{report_id}_{uuid.uuid4().hex[:6]}.pdf"
            generate_pdf_report(fname, summary, qna, last.get("avg_score"))
            with open(fname, "rb") as f:
                st.download_button("Download PDF Report", f.read(), file_name=fname, mime="application/pdf")
            st.success("PDF report generated.")

st.markdown("---")
st.markdown("<small>Developed by Ritam Naskar — AI Resume & Interview Trainer</small>", unsafe_allow_html=True)



# import os
# import streamlit as st
# from dotenv import load_dotenv
# from PIL import Image
# import google.generativeai as genai
# from pdf2image import convert_from_path
# import pytesseract
# import pdfplumber

# # Load environment variables
# load_dotenv()

# # Configure Google Gemini AI
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     try:
#         # Try direct text extraction
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text

#         if text.strip():
#             return text.strip()
#     except Exception as e:
#         print(f"Direct text extraction failed: {e}")

#     # Fallback to OCR for image-based PDFs
#     print("Falling back to OCR for image-based PDF.")
#     try:
#         images = convert_from_path(pdf_path)
#         for image in images:
#             page_text = pytesseract.image_to_string(image)
#             text += page_text + "\n"
#     except Exception as e:
#         print(f"OCR failed: {e}")

#     return text.strip()

# # Function to get response from Gemini AI
# def analyze_resume(resume_text, job_description=None):
#     if not resume_text:
#         return {"error": "Resume text is required for analysis."}
    
#     model = genai.GenerativeModel("gemini-1.5-flash")
    
#     base_prompt = f"""
#     You are an experienced HR with Technical Experience in the field of any one job role from Data Science, Data Analyst, DevOPS, Machine Learning Engineer, Prompt Engineer, AI Engineer, Full Stack Web Development, Big Data Engineering, Marketing Analyst, Human Resource Manager, Software Developer your task is to review the provided resume.
#     Please share your professional evaluation on whether the candidate's profile aligns with the role.ALso mention Skills he already have and siggest some skills to imorve his resume , alos suggest some course he might take to improve the skills.Highlight the strengths and weaknesses.

#     Resume:
#     {resume_text}
#     """

#     if job_description:
#         base_prompt += f"""
#         Additionally, compare this resume to the following job description:
        
#         Job Description:
#         {job_description}
        
#         Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
#         """

#     response = model.generate_content(base_prompt)

#     analysis = response.text.strip()
#     return analysis


# # Streamlit app

# st.set_page_config(page_title=" AI Resume Analyzer Expert ", layout="wide")
# # Title
# st.title("AI Resume/CV Analyzer Expert")
# st.write("Analyze your resume and match it with job descriptions using Google Gemini AI.")

# col1 , col2 = st.columns(2)
# with col1:
#     uploaded_file = st.file_uploader("Upload your resume in (PDF) format: ", type=["pdf"])
# with col2:
#     job_description = st.text_area("Enter Job Description of the role you are applying for: ", placeholder="Paste the job description here...")

# if uploaded_file is not None:
#     st.success("Resume uploaded successfully!")
# else:
#     st.warning("Please upload a resume in PDF format.")


# st.markdown("<div style= 'padding-top: 10px;'></div>", unsafe_allow_html=True)
# if uploaded_file:
#     # Save uploaded file locally for processing
#     with open("uploaded_resume.pdf", "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     # Extract text from PDF
#     resume_text = extract_text_from_pdf("uploaded_resume.pdf")

#     if st.button("Analyze Resume"):
#         with st.spinner("Analyzing resume..."):
#             try:
#                 # Analyze resume
#                 analysis = analyze_resume(resume_text, job_description)
#                 st.success("Analysis complete!")
#                 st.write(analysis)
#             except Exception as e:
#                 st.error(f"Analysis failed: {e}")

# #Footer
# st.markdown("---")
# st.markdown("""<p style= 'text-align: center;' >Powered by <b>Streamlit</b> and <b>Google Gemini AI</b> | Developed by <a href="https://www.linkedin.com/in/ritam-naskar-700b88294"  target="_blank" style='text-decoration: none; color: #FFFFFF'><b>Ritam Naskar</b></a></p>""", unsafe_allow_html=True)
