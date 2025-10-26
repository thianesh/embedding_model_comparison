import json
import os
import streamlit as st
from embedding_models.models_conf import models_to_use

st.set_page_config(page_title="RAG Comparison", layout="wide")

# Loading temp conf
TEMP_FILE = "models_temp.json"

# Load models from temp file if exists
if os.path.exists(TEMP_FILE):
    with open(TEMP_FILE, "r") as f:
        st.session_state.table_data = json.load(f)
elif "table_data" not in st.session_state:
    st.session_state.table_data = models_to_use

pg = st.navigation([
        "pages/1_models.py", 
        "pages/2_upload.py",
        "pages/3_query.py",        
        ]) 

pg.run()
