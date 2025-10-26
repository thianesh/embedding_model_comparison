from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

device = os.getenv("DEVICE")

# pub_med_bert = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
# qwen_3_600m = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
# med_embed_small = SentenceTransformer("abhinand/MedEmbed-small-v0.1")
# embeddinggemma_300m = SentenceTransformer("google/embeddinggemma-300m")
# embeddings = model.encode(["sample text 1", "sample text 2"])
# print(embeddings)  # should show (2, 768)

def get_models(model_names: list):
    models = []
    for model_name in model_names:
        model = SentenceTransformer(model_name, device=device)
        models.append(model)
    models
         
def get_model(model_name: list):
    return SentenceTransformer(model_name, device=device)
