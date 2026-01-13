import fitz  # PyMuPDF
import ollama
import io
from PIL import Image
from pathlib import Path
from typing import List, Dict
from config import MIN_IMAGE_SIZE, VISION_MODEL

def extract_images_from_pdf(pdf_path: Path) -> List[Dict[str, any]]: # type: ignore
    """
    Extract images from PDF using PyMuPDF and generate descriptions using LLaVA
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of dictionaries containing image descriptions and metadata
    """
    image_chunks = []
    
    try:
        # Open PDF with PyMuPDF (faster than pdf2image)
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Get images from page
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                try:
                    # Extract image
                    xref = img_info[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image to check dimensions
                    image = Image.open(io.BytesIO(image_bytes))
                    width, height = image.size
                    
                    # Skip small images (icons, logos)
                    if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                        continue
                    
                    # Generate description using LLaVA
                    description = describe_image_with_llava(image_bytes)
                    
                    if description:
                        image_chunks.append({
                            "text": description,
                            "page": page_num + 1,
                            "type": "image",
                            "image_index": img_index,
                            "dimensions": f"{width}x{height}"
                        })
                
                except Exception as e:
                    print(f"Warning: Could not process image {img_index} on page {page_num + 1}: {e}")
                    continue
        
        pdf_document.close()
    
    except Exception as e:
        print(f"Error extracting images from {pdf_path.name}: {e}")
        raise
    
    return image_chunks

def describe_image_with_llava(image_bytes: bytes) -> str:
    """
    Use LLaVA to generate a description of an image
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        Text description of the image
    """
    try:
        # Specialized prompt for troubleshooting images
        prompt = """Describe this troubleshooting image in detail. Include:
- What type of screen, dialog, or interface is shown
- Any error messages, error codes, or status indicators visible
- UI elements like buttons, icons, menus, or indicators
- What problem or solution is being illustrated
- Any text visible in the image
Keep the description concise but include all technical details that would help someone understand this troubleshooting step."""
        
        # Call LLaVA vision model
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_bytes]
            }]
        )
        
        description = response['message']['content']
        
        # Add prefix to indicate this is from an image
        return f"[IMAGE DESCRIPTION] {description}"
    
    except Exception as e:
        print(f"Error generating image description with LLaVA: {e}")
        # Return a basic description if LLaVA fails
        return "[IMAGE] Troubleshooting screenshot (description unavailable)"

def extract_images_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict[str, any]]: # type: ignore
    """
    Extract images from PDF bytes (for API uploads)
    
    Args:
        pdf_bytes: PDF file content as bytes
        
    Returns:
        List of dictionaries containing image descriptions and metadata
    """
    image_chunks = []
    
    try:
        # Open PDF from bytes
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Get images from page
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                try:
                    # Extract image
                    xref = img_info[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image to check dimensions
                    image = Image.open(io.BytesIO(image_bytes))
                    width, height = image.size
                    
                    # Skip small images
                    if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                        continue
                    
                    # Generate description
                    description = describe_image_with_llava(image_bytes)
                    
                    if description:
                        image_chunks.append({
                            "text": description,
                            "page": page_num + 1,
                            "type": "image",
                            "image_index": img_index,
                            "dimensions": f"{width}x{height}"
                        })
                
                except Exception as e:
                    print(f"Warning: Could not process image {img_index} on page {page_num + 1}: {e}")
                    continue
        
        pdf_document.close()
    
    except Exception as e:
        print(f"Error extracting images from PDF bytes: {e}")
        raise
    
    return image_chunks