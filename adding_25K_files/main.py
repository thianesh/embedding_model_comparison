embedding_token_length = 500
overlap_token = 50

from ..util.token_chunker import token_level_chunks

model_name = "Qwen/Qwen3-Embedding-0.6B"



import os
import json
from pathlib import Path

# Directory containing text files
BASE_DIR = Path("/path/to/your/text_files")
PROGRESS_FILE = Path("progress.json")

def load_progress():
    """Load processed file list."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_progress(processed_files):
    """Save progress periodically."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(processed_files), f, indent=2)

def read_text_file(file_path):
    """Safely read a text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None

def process_file(file_path, processed_files):
    """Process each text file (customize this)."""
    content = read_text_file(file_path)
    if content is not None:
        # Example: just print length (replace this with your logic)
        print(f"✅ Read {file_path} ({len(content)} chars)")

        # Chunking & Embedding and storing embeddings
        chunks = token_level_chunks(
            text=content,
            model_name=model_name,
            chunk_size=embedding_token_length,
            chunk_overlap=overlap_token,
            add_special_tokens=False,
        )

        # Embedding chunks

    processed_files.add(str(file_path))

def main():
    processed_files = load_progress()

    all_files = sorted(BASE_DIR.rglob("*.txt"))  # recursive
    print(f"Found {len(all_files)} text files.")

    for file_path in all_files:
        if str(file_path) in processed_files:
            continue  # Skip processed files

        process_file(file_path, processed_files)

        # Save progress after each file (or every N files for speed)
        save_progress(processed_files)

    print("🎉 All files processed successfully!")

if __name__ == "__main__":
    main()
