import PyPDF2
import io
from typing import List, Dict
from pathlib import Path
from config import CHUNK_SIZE

def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, any]]: # type: ignore
    """
    Extract text from PDF and split into chunks
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of dictionaries containing text chunks and metadata
    """
    chunks = []
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    
                    if not text or not text.strip():
                        continue
                    
                    # Split into smaller chunks
                    for i in range(0, len(text), CHUNK_SIZE):
                        chunk = text[i:i + CHUNK_SIZE]
                        if chunk.strip():
                            chunks.append({
                                "text": chunk.strip(),
                                "page": page_num + 1,
                                "type": "text"
                            })
                
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error reading PDF {pdf_path.name}: {e}")
        raise
    
    return chunks

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict[str, any]]: # type: ignore
    """
    Extract text from PDF bytes (for API uploads)
    
    Args:
        pdf_bytes: PDF file content as bytes
        
    Returns:
        List of dictionaries containing text chunks and metadata
    """
    chunks = []
    
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                text = page.extract_text()
                
                if not text or not text.strip():
                    continue
                
                # Split into smaller chunks
                for i in range(0, len(text), CHUNK_SIZE):
                    chunk = text[i:i + CHUNK_SIZE]
                    if chunk.strip():
                        chunks.append({
                            "text": chunk.strip(),
                            "page": page_num + 1,
                            "type": "text"
                        })
            
            except Exception as e:
                print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                continue
    
    except Exception as e:
        print(f"Error reading PDF bytes: {e}")
        raise
    
    return chunks