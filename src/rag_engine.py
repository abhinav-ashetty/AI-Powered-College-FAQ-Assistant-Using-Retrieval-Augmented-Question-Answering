import os
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import FAISS_INDEX_DIRECTORY, EMBEDDING_MODEL_NAME, GEMINI_API_KEY

if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

class CollegeRAGEngine:
    def __init__(self):
        print(f"[*] Initializing HuggingFace embeddings ({EMBEDDING_MODEL_NAME})...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        print(f"[*] Loading FAISS vector store from '{FAISS_INDEX_DIRECTORY}'...")
        if not os.path.exists(FAISS_INDEX_DIRECTORY):
            raise FileNotFoundError(
                f"FAISS index not found at '{FAISS_INDEX_DIRECTORY}'. Run 'python -m src.ingestion' first."
            )
            
        self.vectorstore = FAISS.load_local(
            FAISS_INDEX_DIRECTORY,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )
        
        # Candidate models prioritized by stability and quota availability
        self.candidate_models = [
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-3.8-flash"
        ]

    def _call_gemini_with_fallback(self, prompt: str) -> str:
        last_error = None
        for model_name in self.candidate_models:
            print(f"[3/4] Attempting generation with '{model_name}'...")
            for attempt in range(2):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        temperature=0.1,
                        request_timeout=60.0,
                        max_retries=1
                    )
                    response = llm.invoke(prompt)
                    print(f"[4/4] Successfully generated response via '{model_name}'.")

                    if isinstance(response.content, list):
                        return "".join(
                            part.get("text", str(part)) if isinstance(part, dict) else str(part)
                            for part in response.content
                        ).strip()
                    return str(response.content).strip()

                except Exception as e:
                    err_str = str(e)
                    print(f"[!] '{model_name}' attempt {attempt + 1} failed: {type(e).__name__} - {err_str}")
                    last_error = e
                    if "503" in err_str or "504" in err_str or "UNAVAILABLE" in err_str:
                        time.sleep(2)
                    else:
                        break

        return f"Error communicating with AI service: All candidate models unavailable. Details: {last_error}"

    def query(self, question: str, department: str = "All", semester: str = "All") -> dict:
        print(f"\n[1/4] Received question: {question} (Dept: {department}, Sem: {semester})")
        
        print("[2/4] Retrieving documents from FAISS...")
        matched_docs = self.retriever.invoke(question)
        print(f"      Found {len(matched_docs)} matching documents.")
        
        context_parts = []
        for i, doc in enumerate(matched_docs, 1):
            src = doc.metadata.get("source", "Unknown")
            context_parts.append(f"--- Document {i} (Source: {src}) ---\n{doc.page_content}")
            
        context_str = "\n\n".join(context_parts) if context_parts else "No relevant syllabus records found."
        
        prompt = (
            "You are the official College Academic FAQ Assistant. Answer the student's question strictly using the provided context snippets below.\n"
            "If the information is not present in the context, explicitly state:\n"
            "\"This information is not available in the current college records. Please verify directly with the respective departmental coordinator or administrative office.\"\n"
            "Do not speculate or extrapolate. Always mention the source, module, or course code if identifiable.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question:\n{question}\n\n"
            "Helpful & Accurate Answer:"
        )
        
        answer_text = self._call_gemini_with_fallback(prompt)
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in matched_docs]))
        
        return {
            "answer": answer_text,
            "sources": sources,
            "matched_chunks": len(matched_docs)
        }
