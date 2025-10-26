from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()
device = os.getenv("DEVICE")

models = [
        {"model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", "bucket": "vector-db-storage-exp", "index": "5999368648727199744"},
        {"model": "Qwen/Qwen3-Embedding-0.6B", "bucket": "my-bucket-2", "index": "bio-index"},
        {"model": "abhinand/MedEmbed-small-v0.1", "bucket": "my-bucket-3", "index": "st-index"},
        {"model": "google/embeddinggemma-300m", "bucket": "my-bucket-4", "index": "scibert-index"},
    ]

for model in models:
    embed_model = SentenceTransformer(model['model'], device=device)
    embeddings = embed_model.encode(["sample text 1", "sample text 2"])
    print(f"Shape for model: {model['model']} ")
    print(embeddings.shape)
    del embed_model
