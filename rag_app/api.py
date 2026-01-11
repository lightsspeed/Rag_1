from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from chromadb.config import Settings
import ollama
import PyPDF2
import io
import hashlib

app = FastAPI(title="RAG API Server")

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
    persist_directory="./chroma_db",
    anonymized_telemetry=False
))

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name="pdf_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

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

def extract_text_from_pdf(pdf_file: bytes) -> List[dict]:
    """Extract text from PDF and split into chunks"""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
    chunks = []
    
    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        # Split into smaller chunks (roughly 500 characters)
        chunk_size = 500
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "page": page_num + 1
                })
    
    return chunks

def generate_embedding(text: str) -> List[float]:
    """Generate embeddings using Ollama"""
    try:
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        
        if "embedding" not in response:
            raise ValueError("Ollama response missing 'embedding' field")
        
        embedding = response["embedding"]
        
        if not embedding or len(embedding) == 0:
            raise ValueError("Ollama returned empty embedding")
        
        return embedding
    except Exception as e:
        error_msg = f"Error generating embedding: {str(e)}"
        print(error_msg)
        raise ValueError(error_msg)

def format_chat_history(chat_history: List[Message], max_messages: int = 6) -> str:
    """Format chat history for context, keeping only recent messages"""
    if not chat_history:
        return ""
    
    # Keep only the last N messages to avoid context overflow
    recent_history = chat_history[-max_messages:]
    
    formatted = "\n\nPrevious conversation:\n"
    for msg in recent_history:
        role = "Human" if msg.role == "user" else "Assistant"
        formatted += f"{role}: {msg.content}\n"
    
    return formatted

@app.post("/upload-pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload and process multiple PDF files"""
    try:
        total_chunks = 0
        processed_files = 0
        
        for file in files:
            if not file.filename or not file.filename.endswith('.pdf'):
                continue
                
            pdf_content = await file.read()
            file_hash = hashlib.md5(pdf_content).hexdigest()
            
            # Extract text chunks
            chunks = extract_text_from_pdf(pdf_content)
            
            if not chunks:
                raise HTTPException(
                    status_code=400, 
                    detail=f"No text could be extracted from {file.filename}. The PDF might be image-based or corrupted."
                )
            
            # Prepare data for ChromaDB
            documents = []
            embeddings = []
            ids = []
            metadatas = []
            
            for idx, chunk in enumerate(chunks):
                try:
                    doc_id = f"{file_hash}_{idx}"
                    chunk_text = chunk["text"].strip()
                    
                    # Skip empty chunks
                    if not chunk_text:
                        continue
                    
                    # Generate embedding
                    embedding = generate_embedding(chunk_text)
                    
                    # Validate embedding is not empty
                    if not embedding or len(embedding) == 0:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to generate embedding for chunk {idx} in {file.filename}. Check if Ollama is running and the 'nomic-embed-text' model is available."
                        )
                    
                    documents.append(chunk_text)
                    embeddings.append(embedding)
                    ids.append(doc_id)
                    metadatas.append({
                        "filename": file.filename,
                        "page": chunk["page"]
                    })
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error processing chunk {idx} in {file.filename}: {str(e)}"
                    )
            
            # Only add to ChromaDB if we have valid data
            if documents and embeddings and len(documents) > 0 and len(embeddings) > 0:
                collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    ids=ids,
                    metadatas=metadatas
                )
                total_chunks += len(documents)
                processed_files += 1
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"No valid chunks were created for {file.filename}"
                )
        
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
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base with conversational context"""
    try:
        # Generate embedding for the question
        question_embedding = generate_embedding(request.question)
        
        # Search in ChromaDB
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=request.top_k
        )
        
        # Validate results
        documents_list = results.get('documents') if results else None
        if not documents_list or not documents_list[0]:
            return QueryResponse(
                answer="I don't have enough information to answer this question.",
                sources=[]
            )
        
        # Prepare context from retrieved documents
        documents = documents_list[0]
        metadatas_list = results.get('metadatas') if results else None
        metadatas = metadatas_list[0] if metadatas_list and metadatas_list[0] else []
        
        context = "\n\n".join(documents)
        sources = [
            f"{m['filename']} (Page {m['page']})" 
            for m in metadatas if m and 'filename' in m and 'page' in m
        ]
        
        # Format chat history for conversational context
        chat_history = request.chat_history or []
        history_context = format_chat_history(chat_history)
        
        # Generate answer using Ollama with conversational context
        prompt = f"""You are a helpful AI assistant answering questions based on provided documents. Use the conversation history to provide contextual answers.

{history_context}

Context from documents:
{context}

Current question: {request.question}

Instructions:
- Answer based on the context provided
- If referring to previous conversation, acknowledge it naturally
- If the answer isn't in the context, say so
- Be conversational and helpful
- If the user asks follow-up questions like "tell me more" or "what about that", refer to the conversation history

Answer:"""
        
        response = ollama.generate(
            model="llama3.2",  # or any model you have pulled
            prompt=prompt
        )
        
        return QueryResponse(
            answer=response['response'],
            sources=sources
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear-database")
async def clear_database():
    """Clear all documents from the knowledge base"""
    try:
        chroma_client.delete_collection("pdf_knowledge_base")
        global collection
        collection = chroma_client.get_or_create_collection(
            name="pdf_knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        return {"message": "Database cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "collection_count": collection.count()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)