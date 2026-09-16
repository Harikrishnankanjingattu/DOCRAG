"""
FastAPI Server for Document RAG with Live Pipeline Demonstration.
Serves RAG API and mounts the monochrome frontend.
"""

import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.rag_engine import RAGEngine
from backend.samples import SAMPLE_DOCUMENTS

app = FastAPI(
    title="Document RAG Interactive Studio",
    description="Full-pipeline Document RAG with live chunking, vector embedding inspection, and vector database similarity comparison.",
    version="1.0.0"
)

# Enable CORS for flexible development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
rag_engine = RAGEngine()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
if not FRONTEND_DIR.exists():
    FRONTEND_DIR = Path.cwd() / "frontend"


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    api_provider: Optional[str] = None
    api_key: Optional[str] = None


class SampleRequest(BaseModel):
    sample_key: str
    chunk_size: Optional[int] = 400
    chunk_overlap: Optional[int] = 80
    api_provider: Optional[str] = None
    api_key: Optional[str] = None


@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "has_document": len(rag_engine.vector_store.chunks) > 0,
        "indexed_chunks": len(rag_engine.vector_store.chunks),
        "document_name": rag_engine.vector_store.doc_name
    }


@app.get("/api/samples")
async def get_samples():
    """Returns list of pre-bundled sample documents for quick testing."""
    return {
        "samples": [
            {
                "key": key,
                "title": data["title"],
                "description": data["description"],
                "char_count": len(data["content"])
            }
            for key, data in SAMPLE_DOCUMENTS.items()
        ]
    }


@app.post("/api/sample")
async def load_sample(req: SampleRequest):
    """Loads a pre-bundled sample document and runs the complete ingestion pipeline."""
    if req.sample_key not in SAMPLE_DOCUMENTS:
        raise HTTPException(status_code=404, detail="Sample document not found.")

    sample = SAMPLE_DOCUMENTS[req.sample_key]
    result = await rag_engine.ingest_document(
        filename=sample["title"],
        content=sample["content"],
        chunk_size=req.chunk_size or 400,
        chunk_overlap=req.chunk_overlap or 80,
        api_provider=req.api_provider,
        api_key=req.api_key
    )
    return result


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(400),
    chunk_overlap: int = Form(80),
    api_provider: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None)
):
    """
    Ingests an uploaded document (PDF, TXT, MD, CSV, etc.),
    chunks it, computes embeddings, and stores it in the vector store.
    """
    try:
        content_bytes = await file.read()
        extracted_text = rag_engine.extract_text(file.filename, content_bytes)
        
        if not extracted_text or not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract readable text from the uploaded file. Please ensure it is not empty or encrypted."
            )

        result = await rag_engine.ingest_document(
            filename=file.filename,
            content=extracted_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            api_provider=api_provider,
            api_key=api_key
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query_document(req: QueryRequest):
    """
    Takes user question, computes query embedding, runs vector DB similarity
    comparison across all chunks, and generates a grounded response with citations.
    """
    if not rag_engine.vector_store.chunks:
        raise HTTPException(status_code=400, detail="No document has been ingested yet.")

    result = await rag_engine.answer_query(
        query=req.query,
        top_k=req.top_k or 3,
        api_provider=req.api_provider,
        api_key=req.api_key
    )
    return result


@app.get("/api/chunks")
async def get_chunks():
    """Inspects all currently stored chunks and their vector information."""
    return {
        "document_name": rag_engine.vector_store.doc_name,
        "total_chunks": len(rag_engine.vector_store.chunks),
        "chunks": [c.to_dict(include_vector_preview=True) for c in rag_engine.vector_store.chunks]
    }


@app.post("/api/reset")
async def reset_document():
    """Clears indexed document and resets the vector store."""
    rag_engine.vector_store.clear()
    return {"status": "cleared"}


# Mount static frontend files if directory exists
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Document RAG server on http://localhost:{port}")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
