import requests
from typing import List, Dict, Optional
from config import BRAVE_API_KEY, TAVILY_API_KEY, WEB_SEARCH_PROVIDER

def search_brave(query: str, num_results: int = 3) -> Dict[str, any]:# type: ignore
    """
    Search using Brave Search API
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        Dictionary containing search results and metadata
    """
    if not BRAVE_API_KEY:
        raise ValueError("Brave API key not configured. Set BRAVE_API_KEY in config.py or environment.")
    
    try:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_API_KEY
        }
        params = {
            "q": query,
            "count": num_results
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract results
        results = []
        sources = []
        
        if "web" in data and "results" in data["web"]:
            for item in data["web"]["results"][:num_results]:
                title = item.get("title", "")
                description = item.get("description", "")
                url = item.get("url", "")
                
                results.append({
                    "title": title,
                    "description": description,
                    "url": url
                })
                sources.append(url)
        
        # Format context for LLM
        context = "\n\n".join([
            f"Title: {r['title']}\nDescription: {r['description']}\nURL: {r['url']}"
            for r in results
        ])
        
        return {
            "context": context,
            "sources": sources,
            "results": results
        }
    
    except Exception as e:
        print(f"Error searching with Brave: {e}")
        return {"context": "", "sources": [], "results": []}

def search_tavily(query: str, num_results: int = 3) -> Dict[str, any]:# type: ignore
    """
    Search using Tavily AI Search API
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        Dictionary containing search results and metadata
    """
    if not TAVILY_API_KEY:
        raise ValueError("Tavily API key not configured. Set TAVILY_API_KEY in config.py or environment.")
    
    try:
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=TAVILY_API_KEY)
        
        # Tavily search
        response = client.search(
            query=query,
            max_results=num_results,
            include_answer=False  # We'll generate our own answer
        )
        
        results = []
        sources = []
        
        for item in response.get("results", [])[:num_results]:
            title = item.get("title", "")
            content = item.get("content", "")
            url = item.get("url", "")
            
            results.append({
                "title": title,
                "description": content,
                "url": url
            })
            sources.append(url)
        
        # Format context for LLM
        context = "\n\n".join([
            f"Title: {r['title']}\nContent: {r['description']}\nURL: {r['url']}"
            for r in results
        ])
        
        return {
            "context": context,
            "sources": sources,
            "results": results
        }
    
    except Exception as e:
        print(f"Error searching with Tavily: {e}")
        return {"context": "", "sources": [], "results": []}

def search_web(query: str, num_results: int = 3, provider: Optional[str] = None) -> Dict[str, any]:# type: ignore
    """
    Search the web using configured provider
    
    Args:
        query: Search query
        num_results: Number of results to return
        provider: Override provider ("brave" or "tavily")
        
    Returns:
        Dictionary containing search results and metadata
    """
    search_provider = provider or WEB_SEARCH_PROVIDER
    
    if search_provider == "brave":
        return search_brave(query, num_results)
    elif search_provider == "tavily":
        return search_tavily(query, num_results)
    else:
        print("Warning: No web search provider configured. Skipping web search.")
        return {"context": "", "sources": [], "results": []}

def should_use_web_search(similarity_score: float, threshold: float = 0.7) -> bool:
    """
    Determine if web search should be used based on similarity score
    
    Args:
        similarity_score: ChromaDB similarity score (0-1, lower is better)
        threshold: Threshold below which to use web search
        
    Returns:
        True if web search should be used
    """
    # In ChromaDB, distance scores closer to 0 = more similar
    # If score > threshold, results are not very relevant, use web search
    return similarity_score > threshold