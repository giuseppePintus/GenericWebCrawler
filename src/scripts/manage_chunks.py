import argparse
import os
import json
from pathlib import Path

def process_chunks(site_path):
    """Process and organize chunks for a given site"""
    pages_dir = Path(site_path) / "pages"
    
    # Create consolidated chunks directory
    chunks_dir = Path(site_path) / "processed_chunks"
    chunks_dir.mkdir(exist_ok=True)
    
    for page_dir in pages_dir.glob("*"):
        if not page_dir.is_dir():
            continue
            
        # Read metadata
        with open(page_dir / "metadata.json") as f:
            metadata = json.load(f)
            
        # Process chunks
        chunks_path = page_dir / "chunks"
        if chunks_path.exists():
            for chunk_file in chunks_path.glob("chunk_*.txt"):
                with open(chunk_file) as f:
                    content = f.read()
                    
                # Create processed chunk with metadata
                chunk_data = {
                    "content": content,
                    "source_url": metadata["url"],
                    "timestamp": metadata["timestamp"]
                }
                
                # Save processed chunk
                output_file = chunks_dir / f"{page_dir.name}_{chunk_file.name}"
                with open(output_file, 'w') as f:
                    json.dump(chunk_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    args = parser.parse_args()
    
    site_path = f"output/{args.site}"
    if not os.path.exists(site_path):
        print(f"Site directory not found: {site_path}")
        return
        
    process_chunks(site_path)

if __name__ == "__main__":
    main()