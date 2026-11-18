# 🎓 College Academic FAQ RAG System

A production-ready Retrieval-Augmented Generation (RAG) assistant designed to answer college syllabus, examination regulations, Continuous Internal Evaluation (CIE), and Semester End Examination (SEE) queries.

---

## 📌 Architecture Overview

```text
       User Query (Browser)
                │
                ▼
      Streamlit Frontend (:8501)
                │  HTTP POST /api/ask
                ▼
       FastAPI Backend (:8000)
                │
                ├───► HuggingFace Embeddings (all-MiniLM-L6-v2)
                │            │ Vector Match
                │            ▼
                ├───► FAISS Vector Index (Local ./faiss_index)
                │            │ Top-k Chunks + Sources
                │            ▼
                └───► Google Gemini API (gemini-2.5-flash / gemini-1.5-flash)
                             │
                             ▼
                     Grounded Response
```

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites
- Python 3.10+ installed
- Google Gemini API key from Google AI Studio (https://aistudio.google.com/app/apikey)

### 2. Configure Environment
Copy `.env.example` to `.env` and add your API key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GEMINI_API_KEY="AIzaSy...your_gemini_api_key..."
FAISS_INDEX_DIRECTORY="./faiss_index"
EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"
```

### 3. Install Dependencies
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. Ingest Documents & Build FAISS Vector Database
Place your PDF syllabi into `data/raw/` (e.g., `ISE_Sem7_CCBDA.pdf`) and run:
```powershell
python -m src.ingestion
```

### 5. Launch Application

**Terminal 1 - FastAPI Backend:**
```powershell
uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 - Streamlit Frontend:**
```powershell
streamlit run frontend.py
```

Open `http://localhost:8501` in your browser.
