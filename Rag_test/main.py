from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from chromadb.config import Settings
import ollama
import hashlib
import logging
from pathlib import Path

# Import utilities
from config import (
    CHROMA_DIR, COLLECTION_NAME, COLLECTION_METADATA,
    LLM_MODEL, MAX_CHAT_HISTORY, WEB_SEARCH_ENABLED,
    SIMILARITY_THRESHOLD, PDF_DIR
)
from utils.embeddings import generate_embedding
from utils.pdf_processor import extract_text_from_pdf_bytes
from utils.image_extractor import extract_images_from_pdf_bytes
from utils.web_search import search_web, should_use_web_search
from utils.file_tracker import FileTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API Server with Image & Web Search")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB
chroma_client = chromadb.Client(Settings(
    persist_directory=str(CHROMA_DIR),
    anonymized_telemetry=False
))

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata=COLLECTION_METADATA
)

# Initialize file tracker
file_tracker = FileTracker()

# Processing status tracking
processing_status = {
    "is_processing": False,
    "current": 0,
    "total": 0,
    "current_file": ""
}

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    chat_history: Optional[List[Message]] = []

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

def format_chat_history(chat_history: List[Message], max_messages: int = MAX_CHAT_HISTORY) -> str:
    """Format chat history for context"""
    if not chat_history:
        return ""
    
    recent_history = chat_history[-max_messages:]
    formatted = "\n\nPrevious conversation:\n"
    for msg in recent_history:
        role = "Human" if msg.role == "user" else "Assistant"
        formatted += f"{role}: {msg.content}\n"
    
    return formatted

def process_single_pdf(pdf_path: Path, file_hash: str) -> int:
    """
    Process a single PDF file (text + images)
    
    Returns:
        Number of chunks added
    """
    logger.info(f"Processing: {pdf_path.name}")
    
    try:
        # Read PDF
        with open(pdf_path, 'rb') as f:
            pdf_content = f.read()
        
        # Extract text chunks
        text_chunks = extract_text_from_pdf_bytes(pdf_content)
        logger.info(f"  - Extracted {len(text_chunks)} text chunks")
        
        # Extract image descriptions
        image_chunks = extract_images_from_pdf_bytes(pdf_content)
        logger.info(f"  - Extracted {len(image_chunks)} images")
        
        # Combine all chunks
        all_chunks = text_chunks + image_chunks
        
        if not all_chunks:
            logger.warning(f"  - No content extracted from {pdf_path.name}")
            return 0
        
        # Prepare data for ChromaDB
        documents = []
        embeddings = []
        ids = []
        metadatas = []
        
        for idx, chunk in enumerate(all_chunks):
            try:
                doc_id = f"{file_hash}_{idx}"
                chunk_text = chunk["text"].strip()
                
                if not chunk_text:
                    continue
                
                # Generate embedding
                embedding = generate_embedding(chunk_text)
                
                if not embedding:
                    continue
                
                documents.append(chunk_text)
                embeddings.append(embedding)
                ids.append(doc_id)
                metadatas.append({
                    "filename": pdf_path.name,
                    "page": chunk["page"],
                    "type": chunk["type"]
                })
            
            except Exception as e:
                logger.error(f"  - Error processing chunk {idx}: {e}")
                continue
        
        # Add to ChromaDB
        if documents and embeddings:
            collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            logger.info(f"  ✓ Added {len(documents)} chunks to database")
            return len(documents)
        
        return 0
    
    except Exception as e:
        logger.error(f"  ✗ Error processing {pdf_path.name}: {e}")
        raise

async def process_local_pdfs_background():
    """Process PDFs from local directory (runs on startup)"""
    global processing_status
    
    try:
        processing_status["is_processing"] = True
        
        # Get unprocessed files
        unprocessed_files = file_tracker.get_unprocessed_files(PDF_DIR)
        
        if not unprocessed_files:
            logger.info("✓ All PDFs already processed")
            processing_status["is_processing"] = False
            return
        
        logger.info(f"Found {len(unprocessed_files)} new/modified PDFs to process")
        
        processing_status["total"] = len(unprocessed_files)
        total_chunks = 0
        
        for i, pdf_path in enumerate(unprocessed_files):
            processing_status["current"] = i + 1
            processing_status["current_file"] = pdf_path.name
            
            try:
                file_hash = file_tracker.calculate_hash(pdf_path)
                chunks_added = process_single_pdf(pdf_path, file_hash)
                total_chunks += chunks_added
                
                # Mark as processed
                file_tracker.mark_processed(pdf_path)
            
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                continue
        
        logger.info(f"✓ Processing complete: {total_chunks} chunks from {len(unprocessed_files)} PDFs")
    
    except Exception as e:
        logger.error(f"Error in background processing: {e}")
    
    finally:
        processing_status["is_processing"] = False
        processing_status["current"] = 0
        processing_status["total"] = 0
        processing_status["current_file"] = ""

@app.on_event("startup")
async def startup_event():
    """Auto-process PDFs on server start"""
    logger.info("🚀 Server starting - checking for new PDFs...")
    await process_local_pdfs_background()
    logger.info("✅ Startup processing complete")

@app.post("/upload-pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload and process PDF files manually"""
    try:
        total_chunks = 0
        processed_files = 0
        
        for file in files:
            if not file.filename or not file.filename.endswith('.pdf'):
                continue
            
            pdf_content = await file.read()
            file_hash = hashlib.md5(pdf_content).hexdigest()
            
            # Extract text chunks
            text_chunks = extract_text_from_pdf_bytes(pdf_content)
            
            # Extract image chunks
            image_chunks = extract_images_from_pdf_bytes(pdf_content)
            
            # Combine
            all_chunks = text_chunks + image_chunks
            
            if not all_chunks:
                raise HTTPException(
                    status_code=400,
                    detail=f"No content extracted from {file.filename}"
                )
            
            # Prepare data
            documents = []
            embeddings = []
            ids = []
            metadatas = []
            
            for idx, chunk in enumerate(all_chunks):
                try:
                    doc_id = f"{file_hash}_{idx}"
                    chunk_text = chunk["text"].strip()
                    
                    if not chunk_text:
                        continue
                    
                    embedding = generate_embedding(chunk_text)
                    
                    if not embedding:
                        continue
                    
                    documents.append(chunk_text)
                    embeddings.append(embedding)
                    ids.append(doc_id)
                    metadatas.append({
                        "filename": file.filename,
                        "page": chunk["page"],
                        "type": chunk["type"]
                    })
                
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    continue
            
            if documents and embeddings:
                collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    ids=ids,
                    metadatas=metadatas
                )
                total_chunks += len(documents)
                processed_files += 1
        
        if processed_files == 0:
            raise HTTPException(
                status_code=400,
                detail="No PDFs were successfully processed"
            )
        
        return {
            "message": f"Successfully processed {processed_files} PDF(s)",
            "total_chunks": total_chunks
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query with intelligent PDF + Web search"""
    try:
        # Generate embedding
        question_embedding = generate_embedding(request.question)
        
        # Search ChromaDB
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=request.top_k
        )
        
        # Extract results
        documents_list = results.get('documents', [[]])
        metadatas_list = results.get('metadatas', [[]])
        distances_list = results.get('distances', [[]])
        
        documents = documents_list[0] if documents_list else []
        metadatas = metadatas_list[0] if metadatas_list else []
        distances = distances_list[0] if distances_list else []
        
        # Prepare PDF context and sources
        pdf_context = "\n\n".join(documents) if documents else ""
        pdf_sources = [
            f"{m['filename']} (Page {m['page']}, {m['type']})"
            for m in metadatas if m
        ]
        
        # Check if we should use web search
        use_web = False
        web_context = ""
        web_sources = []
        
        if WEB_SEARCH_ENABLED and distances:
            avg_distance = sum(distances) / len(distances) if distances else 1.0
            
            if should_use_web_search(avg_distance, SIMILARITY_THRESHOLD) or not documents:
                logger.info(f"Using web search (avg distance: {avg_distance:.3f})")
                use_web = True
                
                try:
                    web_results = search_web(request.question, num_results=3)
                    web_context = web_results.get("context", "")
                    web_sources = web_results.get("sources", [])
                except Exception as e:
                    logger.error(f"Web search failed: {e}")
        
        # Combine contexts
        if pdf_context and web_context:
            combined_context = f"Information from uploaded documents:\n{pdf_context}\n\nAdditional information from web search:\n{web_context}"
        elif web_context:
            combined_context = f"Information from web search:\n{web_context}"
        else:
            combined_context = pdf_context
        
        # Combine sources
        all_sources = pdf_sources + web_sources
        
        # If no context at all
        if not combined_context:
            return QueryResponse(
                answer="I don't have enough information to answer this question.",
                sources=[]
            )
        
        # Format chat history
        chat_history = request.chat_history or []
        history_context = format_chat_history(chat_history)
        
        # Generate answer
        prompt = f"""You are a helpful AI assistant answering questions based on provided information. Use the conversation history to provide contextual answers.

{history_context}

Context:
{combined_context}

Current question: {request.question}

Instructions:
- Answer based on the context provided
- If referring to previous conversation, acknowledge it naturally
- If the answer isn't in the context, say so
- Be conversational and helpful
- Synthesize information from multiple sources when relevant

Answer:"""
        
        response = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt
        )
        
        return QueryResponse(
            answer=response['response'],
            sources=all_sources
        )
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/processing-status")
async def get_processing_status():
    """Get current processing status"""
    return processing_status

@app.delete("/clear-database")
async def clear_database():
    """Clear all documents from the knowledge base"""
    try:
        global collection
        chroma_client.delete_collection(COLLECTION_NAME)
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata=COLLECTION_METADATA
        )
        file_tracker.clear()
        return {"message": "Database cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "collection_count": collection.count(),
        "processing": processing_status["is_processing"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)