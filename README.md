# RAG Knowledge Base (Local)

This repository contains a simple Retrieval-Augmented Generation (RAG) demo:

- FastAPI backend (`rag_app/api.py`) that extracts PDF text, indexes embeddings in ChromaDB, and serves query endpoints.
- Streamlit frontend (`rag_app/app.py`) that uploads PDFs and provides a conversational UI.

**Prerequisites**

- Python 3.10+ installed
- Ollama installed and running locally with required models
- (Optional) Git

**Quick start**

1. Create and activate a virtual environment

Windows (Git Bash / bash):

```bash
python -m venv venv
source venv/Scripts/activate
```

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Windows (CMD):

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

2. Install Python dependencies

```bash
pip install -r rag_app/requirements.txt
```

3. Ensure Ollama is running and models are available

This project uses Ollama for embeddings and generation. Make sure the Ollama daemon is running and the models referenced in the code are pulled locally.

Example (OS-dependent):

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
# start ollama as documented by Ollama (if not already running)
```

4. Run the FastAPI backend

Preferred (development, auto-reload):

```bash
uvicorn rag_app.api:app --reload --host 0.0.0.0 --port 8000
```

Or using the module entrypoint in the file:

```bash
python rag_app/api.py
```

The API serves endpoints on http://localhost:8000 (health, upload, query, clear-database).

Useful API endpoints

- Health: `GET http://localhost:8000/health`
- Upload PDFs: `POST http://localhost:8000/upload-pdfs` (multipart/form-data files)
- Query: `POST http://localhost:8000/query` (JSON payload)
- Clear DB: `DELETE http://localhost:8000/clear-database`

Example query using `curl`:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is RAG?","top_k":3,"chat_history":[]} '
```

5. Run the Streamlit frontend

In a separate terminal (after backend started):

```bash
streamlit run rag_app/app.py
```

This opens a browser UI where you can upload PDFs and chat with the indexed documents.

Project layout

- `rag_app/api.py` — FastAPI server implementation (ChromaDB + Ollama integration)
- `rag_app/app.py` — Streamlit frontend that talks to the API at `http://localhost:8000`
- `rag_app/requirements.txt` — Python dependencies
- `rag_app/chroma_db/` — persistent ChromaDB directory (created at runtime)

Notes & troubleshooting

- Start the backend before starting Streamlit; the frontend expects the API at port 8000.
- If embeddings/generation fail, verify Ollama is running and the models (`nomic-embed-text`, `llama3.2`) are available locally.
- The project persists embeddings under `chroma_db` in the project root; back up or remove if you want to reset indexing.
- To clear the DB programmatically, call the `DELETE /clear-database` endpoint.

If you want, I can also update `rag_app/README.md` with a shorter, app-specific guide. Would you like that?
