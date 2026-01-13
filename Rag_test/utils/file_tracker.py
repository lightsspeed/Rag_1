import json
import hashlib
from pathlib import Path
from typing import Dict, Set
from config import PROCESSED_FILES_JSON

class FileTracker:
    """Track processed PDF files using MD5 hashes"""
    
    def __init__(self, tracking_file: Path = PROCESSED_FILES_JSON):
        self.tracking_file = tracking_file
        self.processed_files: Dict[str, str] = self._load()
    
    def _load(self) -> Dict[str, str]:
        """Load processed files from JSON"""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load tracking file: {e}")
                return {}
        return {}
    
    def _save(self):
        """Save processed files to JSON"""
        try:
            with open(self.tracking_file, 'w') as f:
                json.dump(self.processed_files, f, indent=2)
        except Exception as e:
            print(f"Error saving tracking file: {e}")
    
    @staticmethod
    def calculate_hash(file_path: Path) -> str:
        """Calculate MD5 hash of a file"""
        md5_hash = hashlib.md5()
        with open(file_path, 'rb') as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    
    def is_processed(self, file_path: Path) -> bool:
        """Check if file has been processed and hasn't changed"""
        file_name = file_path.name
        if file_name not in self.processed_files:
            return False
        
        current_hash = self.calculate_hash(file_path)
        return self.processed_files[file_name] == current_hash
    
    def mark_processed(self, file_path: Path):
        """Mark file as processed"""
        file_name = file_path.name
        file_hash = self.calculate_hash(file_path)
        self.processed_files[file_name] = file_hash
        self._save()
    
    def get_unprocessed_files(self, pdf_dir: Path) -> list[Path]:
        """Get list of unprocessed or modified PDF files"""
        unprocessed = []
        
        for pdf_file in pdf_dir.glob("*.pdf"):
            if not self.is_processed(pdf_file):
                unprocessed.append(pdf_file)
        
        return unprocessed
    
    def clear(self):
        """Clear all tracking data"""
        self.processed_files = {}
        self._save()