# 🚀 AI Resume Analyzer & Interview Trainer (RAG-powered)

An **AI-powered career assistant** that helps job seekers prepare for interviews end-to-end.  
This Streamlit-based project combines **Resume Analysis**, **ATS Matching**, **Interview Simulation**, and **Role-specific Question Generation** into a single application.  

---

## 📖 Overview
The **AI Resume Analyzer & Interview Trainer** is designed to act like your **personal HR + Technical mentor**.  
It performs resume evaluation, checks ATS compatibility, generates role-based interview questions using a RAG approach, and allows you to **practice in mock interview mode** with instant AI feedback and scoring.  

This project is ideal for:
- Students preparing for placements
- Professionals switching careers
- Anyone wanting to sharpen interview skills  

---

## ✨ Features

### 📄 Resume Analyzer
- Extracts text from resumes (PDF) using **pdfplumber** (OCR fallback with pytesseract).  
- Evaluates strengths, weaknesses, and missing skills with **Google Gemini AI**.  
- Provides ATS (Applicant Tracking System) keyword match percentage against job descriptions.  
- Suggests **courses & learning resources** to fill skill gaps.  

### 🎤 Interview Trainer
- Role-specific interview questions (Software Developer, Data Scientist, DevOps, HR Manager, etc.).  
- RAG-style retrieval using **SentenceTransformers + FAISS embeddings**.  
- AI-generated **model answers & rubrics** for each question.  

### 🧑‍💻 Mock Interview Mode
- Get a set of technical + HR questions.  
- Submit answers in text (optional: audio STT support).  
- Automatic **scoring (0–100)** using embedding similarity.  
- AI-driven **feedback & improvement tips**.  
- Track performance over time.  

### 📊 Progress Tracking & Reports
- Saves user sessions locally (`user_data/`).  
- Visualizes score improvements across practice sessions.  
- Generates a **professional PDF Interview Readiness Report** with feedback.  

---

## 🛠️ Tech Stack

| Component              | Technology Used |
|------------------------|-----------------|
| **Frontend**           | [Streamlit](https://streamlit.io/) |
| **Backend**            | Python |
| **AI Model**           | [Google Gemini AI](https://ai.google.dev/) |
| **Embeddings (RAG)**   | [SentenceTransformers](https://www.sbert.net/) |
| **Vector Search**      | FAISS |
| **PDF Parsing**        | pdfplumber, pdf2image, pytesseract |
| **Visualization**      | Matplotlib |
| **Report Generation**  | ReportLab |

---

## ⚡ How It Works

1. **Upload Resume (PDF)** → Extract text using pdfplumber/OCR.  
2. **Resume Analyzer** → Gemini AI evaluates, highlights missing skills, and suggests improvements.  
3. **ATS Match** → Compares resume keywords vs job description.  
4. **Interview Trainer** → Retrieves role-based questions, generates model answers.  
5. **Mock Interview** → Candidate answers → AI evaluates → Provides feedback + score.  
6. **Progress & Reports** → Tracks practice history and generates a downloadable PDF report.  

---

## 📂 Project Structure

├── app.py # Main Streamlit application
├── requirements.txt # Project dependencies
├── README.md # Documentation
├── user_data/ # Stores user practice sessions & progress
└── .env # API key (GOOGLE_API_KEY)



---

## 🚀 Setup & Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Ritam200/AI-Resume-Analyzer-Expert
cd AI-Resume-Analyzer-Expert


