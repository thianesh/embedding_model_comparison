import json
import os
import streamlit as st

st.set_page_config(page_title="RAG Comparison", layout="wide")

# Loading temp conf
TEMP_FILE = "models_temp.json"

# # Load models from temp file if exists
# if os.path.exists(TEMP_FILE):
#     with open(TEMP_FILE, "r") as f:
#         st.session_state.table_data = json.load(f)
# elif "table_data" not in st.session_state:
#     st.session_state.table_data = [
#         {"model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", "bucket": "vector-db-storage-exp", "index": "5999368648727199744"},
#         {"model": "Qwen/Qwen3-Embedding-0.6B", "bucket": "my-bucket-2", "index": "bio-index"},
#         {"model": "abhinand/MedEmbed-small-v0.1", "bucket": "my-bucket-3", "index": "st-index"},
#         {"model": "google/embeddinggemma-300m", "bucket": "my-bucket-4", "index": "scibert-index"},
#     ]

pg = st.navigation([
        "pages/1_models.py", 
        "pages/2_upload.py",
        "pages/3_query.py",        
        ]) 

pg.run()
