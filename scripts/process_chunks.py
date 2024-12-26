import argparse
import os
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

def process_chunks(site_path, use_gpu=False):
    """Process chunks with GPU acceleration if available"""
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for processing")
    else:
        device = torch.device("cpu")
        print("Using CPU for processing")

    pages_dir = Path(site_path) / "pages"
    processed_dir = Path(site_path) / "processed"
    processed_dir.mkdir(exist_ok=True)

    # Load model for text processing
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)

    for page_dir in pages_dir.glob("*"):
        if not page_dir.is_dir():
            continue

        # Read page metadata
        with open(page_dir / "metadata.json") as f:
            metadata = json.load(f)

        chunks_dir = page_dir / "chunks"
        if not chunks_dir.exists():
            continue

        # Process each chunk
        processed_chunks = []
        for chunk_file in chunks_dir.glob("chunk_*.txt"):
            with open(chunk_file) as f:
                content = f.read()

            # Process text with GPU if available
            inputs = tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            processed_chunk = {
                "content": content,
                "embedding": embeddings.cpu().numpy().tolist(),
                "source_url": metadata["url"],
                "chunk_id": chunk_file.stem
            }
            processed_chunks.append(processed_chunk)

        # Save processed chunks
        output_file = processed_dir / f"{page_dir.name}_processed.json"
        with open(output_file, 'w') as f:
            json.dump(processed_chunks, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    site_path = f"output/{args.site}"
    if not os.path.exists(site_path):
        print(f"Site directory not found: {site_path}")
        return

    process_chunks(site_path, args.use_gpu)

if __name__ == "__main__":
    main()