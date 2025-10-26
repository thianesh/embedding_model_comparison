import streamlit as st
import json
import os

TEMP_FILE = "models_temp.json"

# Source dropdown + single root-level button (paste this)
if "source" not in st.session_state:
    st.session_state["source"] = "sqlite(local)"

source_option = st.selectbox("Select source", ["sqlite(local)", "vector search(gcp)"], key="source_select")

if st.button("Set Source"):
    st.session_state["source"] = source_option
    st.success(f"Source saved in session: {source_option}")

# Load models from temp file if exists
if os.path.exists(TEMP_FILE):
    with open(TEMP_FILE, "r") as f:
        st.session_state.table_data = json.load(f)
elif "table_data" not in st.session_state:
    st.session_state.table_data = [
        {"model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", "bucket": "vector-db-storage-exp", "index": "5999368648727199744",  "index_endpoint_name":"3063065672146747392", "deployed_index_id":"id_test_vector_search_undeploy_this_if_seen_up" },
        # {"model": "Qwen/Qwen3-Embedding-0.6B", "bucket": "my-bucket-2", "index": "bio-index"},
        # {"model": "abhinand/MedEmbed-small-v0.1", "bucket": "my-bucket-3", "index": "st-index"},
        # {"model": "google/embeddinggemma-300m", "bucket": "my-bucket-4", "index": "scibert-index"},
    ]

def add_row():
    st.session_state.table_data.append({
        "model": "",
        "bucket": "",
        "index": "",
        "index_endpoint_name": "",
        "deployed_index_id": ""
    })

def save_table():
    # Update values from text_inputs
    for i, row in enumerate(st.session_state.table_data):
        row["model"] = st.session_state[f"model_{i}"]
        row["bucket"] = st.session_state[f"bucket_{i}"]
        row["index"] = st.session_state[f"index_{i}"]
        row["index_endpoint_name"] = st.session_state[f"index_endpoint_name_{i}"]
        row["deployed_index_id"] = st.session_state[f"deployed_index_id_{i}"]
    # Save to temp JSON
    with open(TEMP_FILE, "w") as f:
        json.dump(st.session_state.table_data, f, indent=4)
    st.success("Models saved!")

st.write("### Embedding Models Configuration Table")

# Table headers
cols = st.columns([3, 3, 3, 3, 3])
headers = [
    "Embedding Model Name",
    "Google Storage Bucket",
    "Vector Search Index",
    "Index Endpoint Name",
    "Deployed Index ID"
]
for col, header in zip(cols, headers):
    col.markdown(f"**{header}**")

# Table rows
for i, row in enumerate(st.session_state.table_data):
    cols = st.columns([3, 3, 3, 3, 3])
    row["model"] = cols[0].text_input(f"model_{i}", value=row["model"], key=f"model_{i}")
    row["bucket"] = cols[1].text_input(f"bucket_{i}", value=row["bucket"], key=f"bucket_{i}")
    row["index"] = cols[2].text_input(f"index_{i}", value=row["index"], key=f"index_{i}")
    row["index_endpoint_name"] = cols[3].text_input(f"index_endpoint_name_{i}", value=row.get("index_endpoint_name", ""), key=f"index_endpoint_name_{i}")
    row["deployed_index_id"] = cols[4].text_input(f"deployed_index_id_{i}", value=row.get("deployed_index_id", ""), key=f"deployed_index_id_{i}")

st.button("➕ Add Row", on_click=add_row)
st.button("💾 Save Models", on_click=save_table)

st.write("Current table data:", st.session_state.table_data)