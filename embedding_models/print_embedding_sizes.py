from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from models_conf import models_to_use

load_dotenv()
device = os.getenv("DEVICE")
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")  # preferred

from huggingface_hub import snapshot_download
models = models_to_use

for model in models:
    print(f"downloading model: {model['model']} token: {HF_TOKEN}")
    snapshot_download(repo_id=model['model'], repo_type="model", token=HF_TOKEN)

for model in models:
    embed_model = SentenceTransformer(model['model'], device=device)
    embeddings = embed_model.encode(["sample text 1", "sample text 2"])
    print(f"Shape for model: {model['model']} ")
    print(embeddings.shape)
    del embed_model
