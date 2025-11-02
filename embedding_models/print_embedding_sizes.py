from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from models_conf import models_to_use
import time

load_dotenv()
device = os.getenv("DEVICE")
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")  # preferred

from huggingface_hub import snapshot_download
models = models_to_use

# for model in models:
#     print(f"downloading model: {model['model']} token: {HF_TOKEN}")
#     snapshot_download(repo_id=model['model'], repo_type="model", token=HF_TOKEN)

for model in models:
    embed_model = SentenceTransformer(model['model'], device=device, use_auth_token=HF_TOKEN)
    start = time.time()
    embeddings = embed_model.encode(["sample text 1", "sample text 2"])
    print(f"Shape for model: {model['model']} ")
    print(embeddings.shape)
    print(f"Time(Sec) took for embedding in CPU 7950X: {(time.time() - start)/2} per text")
    del embed_model
