from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from src.rag_engine import CollegeRAGEngine

app = FastAPI(title="College Academic FAQ RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None

@app.on_event("startup")
def startup_event():
    global engine
    try:
        engine = CollegeRAGEngine()
        print("[*] CollegeRAGEngine successfully initialized on startup.")
    except Exception as e:
        print(f"[!] Warning on engine init: {e}")

class QueryRequest(BaseModel):
    question: str
    department: Optional[str] = "All"
    semester: Optional[str] = "All"

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "College FAQ RAG Backend"}

@app.post("/api/ask")
def ask(req: QueryRequest):
    global engine
    if engine is None:
        try:
            engine = CollegeRAGEngine()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG Engine not loaded: {e}")
            
    try:
        return engine.query(req.question, req.department, req.semester)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
