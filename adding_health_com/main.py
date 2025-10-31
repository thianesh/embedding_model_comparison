embedding_token_length = 500
overlap_token = 50

import uuid
from embedding_models.all_embedding_models import get_model
from sqlite.sample_query import insert_doc
from util.token_chunker import token_level_chunks

models_to_use =  [
        {"model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", "bucket": "vector-db-storage-exp", "index": "5999368648727199744"},
        {"model": "Qwen/Qwen3-Embedding-0.6B", "bucket": "my-bucket-2", "index": "bio-index"},
        {"model": "abhinand/MedEmbed-small-v0.1", "bucket": "my-bucket-3", "index": "st-index"},
        {"model": "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1", "bucket": "my-bucket-4", "index": "scibert-index"},
        {"model": "google/embeddinggemma-300m", "bucket": "my-bucket-4", "index": "scibert-index"},
        {"model": "BAAI/bge-base-en-v1.5", "bucket": "my-bucket-4", "index": "scibert-index"}
    ]

model_name = models_to_use[5]["model"]  # Choose the first model for this example

import os
import json
from pathlib import Path

# Directory containing text files
BASE_DIR = Path("/home/math/code/article_scraper/health_com_articles")
PROGRESS_FILE = Path("progress_2.json")

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

import numpy as np

def get_batched_embeddings(embed_model, chunks, batch_size=5):
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = embed_model.encode(batch)
        all_embeddings.append(embeddings)

    # Combine all embeddings into a single numpy array
    return np.vstack(all_embeddings)

embed_model = get_model(model_name)

def process_file(file_path, processed_files):
    """Process each text file (customize this)."""
    content = read_text_file(file_path)
    if content is not None:
        # Example: just print length (replace this with your logic)
        print(f"✅ Read {file_path} ({len(content)} chars)")

        print("   - Chunking content...")
        # Chunking & Embedding and storing embeddings
        chunks = token_level_chunks(
            text=content,
            model_name=model_name,
            chunk_size=embedding_token_length,
            chunk_overlap=overlap_token,
            add_special_tokens=False,
        )

        # saving chunks info
        save_file_path = str(file_path).replace('.','')
        with open(f"{save_file_path.replace('health_com_articles', 'health_com_articles_chunks')}_{model_name.replace('/','')}_chunks_info.txt", "w") as info_f:
            info_f.write(f"<total_chunks>{len(chunks)}</total_chunks>\n")
            for i, chunk in enumerate(chunks):
                info_f.write(f"\n<chunks_{i+1}> ---\n")
                info_f.write(chunk + "\n")
                info_f.write(f"\n</chunks_{i+1}> ---\n")

        print(f"   - Created {len(chunks)} chunks.")
        print("   - (Embedding chunks)")
        # Embedding chunks
        embeddings = embed_model.encode(chunks)
        # embeddings = get_batched_embeddings(embed_model, chunks, batch_size=2)

        print("   - Storing embeddings...")
        for chunk, embedding in zip(chunks, embeddings):
            unique_id = str(uuid.uuid4())
            insert_doc(
                doc_id=unique_id,
                content=chunk,
                vec=embedding,
                meta={
                    "title": "Health.com article test", 
                    "source": "from streamlit dev spike oct 25 2025",
                    "text": chunk,
                    "chunk_conf": {
                        "embedding_token_length": embedding_token_length,
                        "overlap_token": overlap_token,
                        "model_name": model_name,
                        "file_path": str(file_path)
                    }
                },
                model=model_name
            )




    processed_files.add(str(file_path))

def start_embedding():
    processed_files = load_progress()

    all_files = sorted(BASE_DIR.rglob("*.md"))  # recursive
    print(f"Found {len(all_files)} text/md files.")

    for index, file_path in enumerate(all_files, start=1):
        if str(file_path) in processed_files:
            continue  # Skip processed files
        print(f"Processing file {index}/{len(all_files)}: {file_path}")
        process_file(file_path, processed_files)

        # Save progress after each file (or every N files for speed)
        save_progress(processed_files)

    print("🎉 All files processed successfully!")
