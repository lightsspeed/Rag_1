import ollama
from typing import List
from config import EMBEDDING_MODEL

def generate_embedding(text: str) -> List[float]:
    """
    Generate embeddings using Ollama
    
    Args:
        text: Text to embed
        
    Returns:
        List of floats representing the embedding
        
    Raises:
        ValueError: If text is empty or embedding generation fails
    """
    try:
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        
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

def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embeddings
    """
    embeddings = []
    for text in texts:
        try:
            embedding = generate_embedding(text)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Warning: Failed to embed text: {str(e)[:100]}...")
            # Skip failed embeddings
            continue
    
    return embeddings