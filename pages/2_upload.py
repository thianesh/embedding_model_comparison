import json
import time
import uuid
from embedding_orchestrator.upload_to_bucket import upload_string_to_bucket
import streamlit as st
from embedding_models.all_embedding_models import get_model, get_models

# ----------------------
# Chunking function
# ----------------------
def chunk_text(text: str, chunk_size: int, chunk_overlap: int, split_by: str = "word"):
    import re

    if split_by == "sentence":
        items = re.split(r'(?<=[.!?])\s+', text)
    else:
        items = text.split()

    chunks = []
    i = 0
    while i < len(items):
        chunk = items[i:i+chunk_size]
        chunks.append(" ".join(chunk) if split_by == "word" else " ".join(chunk))
        i += chunk_size - chunk_overlap
        if i < 0:
            i = 0
            break
    return chunks

# ----------------------
st.write("### Upload / Paste Text and Chunk")

# ----------------------
# 3-column layout
col1, col2, col3 = st.columns([1, 2, 1])  # right column for model cards

# ----------------------
# Left column: input + chunk parameters
with col1:
    uploaded_file = st.file_uploader("Upload a text file", type=["txt", "csv", "json"])
    text_input = st.text_area("Or paste text here", height=200)

    if uploaded_file is not None:
        text_input = uploaded_file.read().decode("utf-8")

    # Chunking parameters
    st.write("### Chunking Parameters")
    chunk_size = st.number_input("Chunk size", min_value=1, value=50, step=1)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, value=10, step=1)
    split_by = st.selectbox("Split by", options=["word", "sentence"])
    
    st.session_state["chunk_conf"] = {
        "chunk_size":  chunk_size,
        "overlap": chunk_overlap,
        "split_by": split_by
    }
    # Chunk button
    if st.button("Chunk Text"):
        if not text_input.strip():
            st.warning("Please provide text or upload a file.")
        else:
            st.session_state["chunks"] = chunk_text(
                text=text_input,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                split_by=split_by
            )

# ----------------------
# Middle column: scrollable chunk preview
with col2:
    chunks = st.session_state.get("chunks", [])
    st.write(f"### Chunk Preview (count: {len(chunks)})")
    if chunks:
        full_text = "\n\n---\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)])
        st.text_area("Chunks Preview", value=full_text, height=600)
    else:
        st.write("Chunks will appear here after clicking 'Chunk Text'.")

# ----------------------
# Right column: model cards
with col3:
    st.write("### Embedding Models")
    # Make sure models exist
    models = st.session_state.get("table_data", [])
    if not models:
        st.info("No models found. Add some models on the Models page.")
    else:
        for i, model in enumerate(models):
            st.markdown(f"**{model['model']}**")
            if st.button(f"Embed, Upload and Index ({model['model']})", key=f"embed_{i}"):
                if not chunks:
                    st.warning("Chunk the text first!")
                else:
                    st.success(f"Started for {model['model']}.")

                    with st.spinner(text="In progress...", show_time=True, width="content"):
                        # for i in range(5):
                        #     time.sleep(0.5)  # simulate work
                        #     st.write(f"Step {i+1}/5 done")  # dynamic message updates
                        
                        st.write(f"(1/5) Loading model: {model['model']}")
                        embed_model = get_model(model['model'])
                        st.write(f"(2/5) Embedding chunks")
                        embeddings = embed_model.encode(chunks)
                        embeddings
                        st.write(f"(3/5) Removing model from memory")
                        del embed_model
                        st.write(f"(4/5) Vector to bucket")
                        
                        ## Creating vector with metadata
                        jsonl_to_save = []
                        i = 1
                        folder_name = f"{int(time.time())}_id_{uuid.uuid4()}"
                        for chunk, embedding in zip(chunks, embeddings):
                            unique_id = str(uuid.uuid4())
                            single_data = {
                                "id": unique_id, 
                                "embedding": [float(x) for x in embedding], 
                                "embedding_metadata": {
                                    "title": "Medembed test", 
                                    "source": "from streamlit dev spike oct 25 2025",
                                    "text": chunk,
                                    "chunk_conf": st.session_state.get('chunk_conf', {})
                                    }
                            }
                            single_data_json = json.dumps(single_data)

                            st.write(f"saving {i}/{len(chunks)} embedding")
                            upload_string_to_bucket(
                                bucket_name=model["bucket"],
                                content=single_data_json,
                                destination_blob_name=f"{model['model'].replace("/", "-")}_time_{folder_name}/data_{unique_id}.json"
                            )
                            i += 1

                        
                        st.write(f"(5/5) Indexing")


                        time.sleep(0.5)


                    st.success("✅ Completed!")