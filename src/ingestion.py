import json
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.config import (
    DATA_RAW_DIR,
    FAQ_JSON_PATH,
    FAISS_INDEX_DIRECTORY,
    EMBEDDING_MODEL_NAME,
)

def parse_filename_metadata(filename: str) -> dict:
    stem = Path(filename).stem
    parts = stem.split("_")
    department = "General"
    semester = "All"
    
    if len(parts) >= 2:
        if parts[0].upper() in ["ISE", "CSE", "ECE", "MECH", "CIVIL"]:
            department = parts[0].upper()
        if "Sem" in parts[1]:
            semester = parts[1].replace("Sem", "Sem ")
            
    return {"department": department, "semester": semester, "source": filename}

def load_raw_pdfs() -> list:
    documents = []
    if not os.path.exists(DATA_RAW_DIR):
        os.makedirs(DATA_RAW_DIR, exist_ok=True)
        return documents
        
    for file_path in Path(DATA_RAW_DIR).glob("*.pdf"):
        meta = parse_filename_metadata(file_path.name)
        try:
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            for page in pages:
                page.metadata.update(meta)
            documents.extend(pages)
            print(f"[+] Loaded {file_path.name} -> Dept: {meta['department']}, Sem: {meta['semester']}")
        except Exception as e:
            print(f"[!] Failed to parse {file_path.name}: {e}")
            
    return documents

def load_faq_json() -> list:
    documents = []
    if not os.path.exists(FAQ_JSON_PATH):
        return documents
        
    try:
        with open(FAQ_JSON_PATH, "r", encoding="utf-8") as f:
            faq_data = json.load(f)
            
        for item in faq_data:
            q = item.get("question", "")
            a = item.get("answer", "")
            content = f"Question: {q}\nAnswer: {a}"
            doc = Document(
                page_content=content,
                metadata={
                    "source": item.get("source", "FAQ Dataset"),
                    "department": item.get("department", "General"),
                    "semester": item.get("semester", "All"),
                    "is_faq": True,
                }
            )
            documents.append(doc)
        print(f"[+] Loaded {len(documents)} structured Q&A pairs from {FAQ_JSON_PATH}")
    except Exception as e:
        print(f"[!] Error loading JSON FAQ dataset: {e}")
        
    return documents

def build_vector_store():
    print("[*] Starting document ingestion pipeline...")
    pdf_docs = load_raw_pdfs()
    faq_docs = load_faq_json()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
    )
    
    split_docs = text_splitter.split_documents(pdf_docs) if pdf_docs else []
    total_docs = split_docs + faq_docs
    
    if not total_docs:
        print("[!] No documents found to ingest. Ensure data/raw/ has PDFs or data/faq_dataset.json exists.")
        return
        
    print(f"[*] Total indexed chunks/entries: {len(total_docs)}")
    print(f"[*] Initializing HuggingFace Embeddings model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print(f"[*] Building FAISS Vector Index at '{FAISS_INDEX_DIRECTORY}'...")
    vectorstore = FAISS.from_documents(total_docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_DIRECTORY)
    print(f"[✓] FAISS index saved successfully to '{FAISS_INDEX_DIRECTORY}'.")

if __name__ == "__main__":
    build_vector_store()
