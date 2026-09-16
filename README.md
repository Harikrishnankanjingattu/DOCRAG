# Document RAG Studio — Live Pipeline & Vector Similarity Visualizer

A fully functional, deployable **Document Retrieval-Augmented Generation (RAG)** application with a minimalist monochrome (white & black) user interface. It demonstrates the entire RAG pipeline in real-time:
1. **Document Ingestion & Parsing** (`.pdf`, `.txt`, `.md`, `.docx`, `.csv`)
2. **Interactive Chunking Breakdown** with sentence boundary preservation and highlighted boundary overlaps
3. **Dense Vector Embeddings Demonstration** (128-D unit vector space, L2 norms, magnitude spectrum sparkbars)
4. **Vector Store Table & Index Inspection**
5. **Interactive Document Chatbot**
6. **Real-time Query Vectorization & Cosine Similarity Ranking Visualizer** across all chunks in the vector database with source citations

---

## 🌟 Key Features

- **No Login / Frictionless**: Jump straight in with instant document upload or 1-click sample documents.
- **Minimalist Monochrome UI**: High-contrast white background (`#ffffff`) and jet-black UI accents (`#000000`), crisp monospace typography, glass borders, and zero clutter.
- **Zero-Dependency Vector Engine**: Runs 100% locally out-of-the-box using pure Python and NumPy for dense semantic vectorization and grounded extractive synthesis.
- **Optional Neural LLM Support**: Switch to Google Gemini (`gemini-1.5-flash` + `text-embedding-004`) or OpenAI (`gpt-4o-mini` + `text-embedding-3-small`) in Pipeline Config settings anytime.
- **Live Vector Math Visualizer**:
  $$\cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{c}}{\|\mathbf{q}\| \|\mathbf{c}\|}$$
  Inspect query vectors, cosine similarity scores, dot products, and rank orders for all stored chunks in real-time.
- **Deployable Anywhere**: Self-contained FastAPI backend serving both API endpoints and the static frontend on a single port. Includes `Dockerfile` for Cloud Run, Render, Railway, HuggingFace Spaces, or Fly.io.

---

## 🚀 Quick Start (Local)

### 1. Install & Run (Windows)
Double-click `run.bat` or run:
```bash
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

### 2. Install & Run (Linux / macOS)
```bash
chmod +x run.sh
./run.sh
```

Open your browser at:
👉 **`http://localhost:8000`**

---

## 🐳 Docker Deployment

Build and run using Docker:
```bash
docker build -t document-rag .
docker run -p 8000:8000 document-rag
```
Or deploy directly to **Render**, **Railway**, **Fly.io**, or **Google Cloud Run** using the included `Dockerfile`.

---

## 📁 Project Structure

```
documentrag/
├── backend/
│   ├── main.py          # FastAPI application & static server
│   ├── rag_engine.py    # Document parser, chunker, vectorizer, cosine search & synthesis
│   └── samples.py       # Pre-bundled technical sample documents
├── frontend/
│   ├── index.html       # Clean monochrome HTML layout
│   ├── css/
│   │   └── style.css    # Minimalist white/black design system
│   └── js/
│       └── app.js       # Dynamic UI state, pipeline renderers, chat & vector visualizer
├── Dockerfile           # Production container configuration
├── requirements.txt     # Python dependencies
├── run.bat              # Windows startup script
├── run.sh               # Unix startup script
└── README.md            # Documentation
```

---

## 📡 API Endpoints

- `GET /api/health`: Server health and active index status.
- `GET /api/samples`: List pre-bundled sample documents.
- `POST /api/sample`: Ingest a sample document (`sample_key`, `chunk_size`, `chunk_overlap`).
- `POST /api/upload`: Multipart upload for custom files (PDF, TXT, MD, DOCX).
- `POST /api/query`: Ask question, returns query embedding, full cosine comparisons ranking, and synthesized answer with citations.
- `GET /api/chunks`: Inspect all chunks and vector coordinates currently in database.
- `POST /api/reset`: Clear vector store.
