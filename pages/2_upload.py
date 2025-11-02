import json
import time
import uuid
from embedding_orchestrator.upload_to_bucket import upload_string_to_bucket
from sqlite.sample_query import insert_doc
import streamlit as st
from embedding_models.all_embedding_models import get_model, get_models

from util.token_chunker import token_level_chunks

def chunk_text(text: str, chunk_size: int, chunk_overlap: int, split_by: str = "word", model_name: str = None):
    """
    If split_by == "token", model_name must be provided (HF model id).
    """
    import re

    if split_by == "sentence":
        items = re.split(r'(?<=[.!?])\s+', text)
        # existing sentence-level chunking (like your original)
        chunks = []
        i = 0
        while i < len(items):
            chunk = items[i:i+chunk_size]
            chunks.append(" ".join(chunk))
            i += chunk_size - chunk_overlap
            if i < 0:
                i = 0
                break
        return chunks

    if split_by == "word":
        items = text.split()
        chunks = []
        i = 0
        while i < len(items):
            chunk = items[i:i+chunk_size]
            chunks.append(" ".join(chunk))
            i += chunk_size - chunk_overlap
            if i < 0:
                i = 0
                break
        return chunks

    if split_by == "token":
        if model_name is None:
            raise ValueError("model_name must be provided for token-level chunking")
        return token_level_chunks(
            text=text,
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_special_tokens=False,
        )

    raise ValueError("split_by must be one of: 'word', 'sentence', 'token'")

# ----------------------
st.write("### Upload / Paste Text and Chunk")

# ----------------------
# 3-column layout
col1, col2, col3 = st.columns([1, 2, 1])  # right column for model cards

# Left column: accept multiple files (max 1000) and chunk per-file
with col1:
    # Allow multiple file uploads, limit to 1000 files
    uploaded_files = st.file_uploader(
        "Upload text files (you can select multiple)", 
        type=["txt", "csv", "json"], 
        accept_multiple_files=True
    )
    text_input = st.text_area("Or paste text here (used only if no files uploaded)", height=200)

    # If files are uploaded, enforce a hard max and show counts
    MAX_FILES = 1000
    if uploaded_files:
        if len(uploaded_files) > MAX_FILES:
            st.warning(f"You uploaded {len(uploaded_files)} files — only the first {MAX_FILES} will be processed.")
            uploaded_files = uploaded_files[:MAX_FILES]

        # Preview uploaded files
        st.write(f"Uploaded {len(uploaded_files)} file(s):")
        for f in uploaded_files:
            st.write(f"- {f.name} ({f.type or 'unknown type'}, size={getattr(f, 'size', 'unknown')} bytes)")

    # Chunking parameters
    st.write("### Chunking Parameters")
    chunk_size = st.number_input("Chunk size", min_value=1, value=50, step=1)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, value=10, step=1)
    split_by = st.selectbox("Split by", options=["word", "sentence", "token"])

    model_name_for_token = None
    if split_by == "token":
        # assume model selection exists on right-hand card; here quick input:
        model_name_for_token = st.text_input("Tokenizer model (HF id)", value="sentence-transformers/all-MiniLM-L6-v2")

    st.session_state["chunk_conf"] = {
        "chunk_size": chunk_size,
        "overlap": chunk_overlap,
        "split_by": split_by
    }

    # Chunk button
    if st.button("Chunk Text"):
        # Validate input: either files or pasted text required
        if (not uploaded_files or len(uploaded_files) == 0) and not text_input.strip():
            st.warning("Please provide text (paste) or upload at least one file.")
        else:
            # If files present, chunk each file separately and keep metadata
            all_chunks = []
            if uploaded_files and len(uploaded_files) > 0:
                # Process files one-by-one to avoid mixing contexts
                st.info("Processing uploaded files...")
                for f in uploaded_files:
                    try:
                        raw = f.read()
                        # decode bytes to str (handle both text and csv/json as utf-8)
                        if isinstance(raw, bytes):
                            text = raw.decode("utf-8", errors="replace")
                        else:
                            text = str(raw)
                    except Exception as e:
                        st.error(f"Failed to read {f.name}: {e}")
                        continue

                    # Optional: prepend filename boundary so chunks keep provenance
                    header = f"\n\n--- FILE: {f.name} ---\n\n"
                    text_with_header = header + text

                    file_chunks = chunk_text(
                        text=text_with_header,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        split_by=split_by,
                        model_name=model_name_for_token
                    )

                    # Keep filename with its chunks
                    all_chunks.append({
                        "filename": f.name,
                        "num_chunks": len(file_chunks),
                        "chunks": file_chunks
                    })
                    st.write(f"→ {f.name}: {len(file_chunks)} chunks")

            else:
                # No files uploaded — chunk the pasted text as single "virtual file"
                st.info("Processing pasted text...")
                text = text_input
                file_chunks = chunk_text(
                    text=text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    split_by=split_by,
                    model_name=model_name_for_token
                )
                all_chunks.append({
                    "filename": "pasted_text",
                    "num_chunks": len(file_chunks),
                    "chunks": file_chunks
                })
                st.write(f"Pasted text: {len(file_chunks)} chunks")

            # Save into session_state for downstream use
            st.session_state["chunks"] = all_chunks
            st.success("Chunking complete.")

    # Download chunks as JSON
    # if st.button("Download Chunks as JSON"):
        chunks = st.session_state.get("chunks", [])
        if not chunks:
            st.warning("No chunks available to download. Please chunk text first.")
        else:
            # Prepare JSON structure
            output_data = {
                "chunk_conf": st.session_state.get("chunk_conf", {}),
                "files": chunks
            }
            json_str = json.dumps(output_data, indent=2)

            # Provide download link
            st.download_button(
                label="Download Chunks JSON",
                data=json_str,
                file_name="chunked_text.json",
                mime="application/json"
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

# --- START REPLACEMENT FOR Right column: model cards ---
with col3:
    st.write("### Embedding Models")
    # Make sure models exist
    models = st.session_state.get("table_data", [])
    if not models:
        st.info("No models found. Add some models on the Models page.")
    else:
        # NEW: batching control
        st.write("### Batch / throughput settings")
        batch_size = st.number_input("Embedding batch size", min_value=1, value=64, step=1, help="Number of chunks to encode per model call")

        # Prepare flattened chunk items from session_state["chunks"]
        # New expected format: list of { "filename":..., "num_chunks": int, "chunks": [ ... ] }
        raw_chunks = st.session_state.get("chunks", [])
        # flattened list of dicts: { filename, file_chunk_count, chunk_index (0-based), text }
        chunk_items = []
        for file_entry in raw_chunks:
            fname = file_entry.get("filename", "unknown")
            file_chunk_count = file_entry.get("num_chunks", len(file_entry.get("chunks", [])))
            for idx, c in enumerate(file_entry.get("chunks", [])):
                chunk_items.append({
                    "filename": fname,
                    "file_chunk_count": file_chunk_count,
                    "chunk_index": idx,
                    "text": c
                })

        total_chunks = len(chunk_items)
        # helper to yield batches
        def batched(iterable, n):
            for i in range(0, len(iterable), n):
                yield iterable[i:i+n]

        # NOTE: chunk_items may be empty; keep previous behaviour
        if st.button(f"Embed, Upload and Index for all", key=f"embed_all"):
            if total_chunks == 0:
                st.warning("Chunk the text first!")
            else:
                st.success(f"Started for all models.")
                with st.spinner(text="In progress...", show_time=True, width="content"):
                    for model in models:
                        st.write(f"**Processing model: {model['model']}**")
                        embed_model = get_model(model['model'])

                        # encode in batches
                        all_embeddings = []
                        processed = 0
                        for batch in batched(chunk_items, batch_size):
                            texts = [item["text"] for item in batch]
                            st.write(f"Embedding batch: {processed+1} -> {processed+len(texts)} / {total_chunks}")
                            embeddings = embed_model.encode(texts)
                            # assume embeddings align with texts order
                            for item, emb in zip(batch, embeddings):
                                all_embeddings.append((item, emb))
                            processed += len(texts)

                        # done with model
                        del embed_model

                        vector_source = st.session_state.get("source") 
                        assert vector_source
                        if vector_source == "sqlite(local)":
                            st.write(f"**Using sqlite - Local**")
                            i = 1
                            for item, embedding in all_embeddings:
                                unique_id = str(uuid.uuid4())
                                st.write(f"saving {i}/{total_chunks} embedding ({item['filename']} #{item['chunk_index']+1}/{item['file_chunk_count']})")
                                insert_doc(
                                    doc_id=unique_id,
                                    content=item["text"],
                                    vec=embedding,
                                    meta={
                                        "title": item.get("filename"),
                                        "source": "from streamlit dev spike",
                                        "text": item["text"],
                                        "filename": item.get("filename"),
                                        "file_chunk_count": item.get("file_chunk_count"),
                                        "chunk_index": item.get("chunk_index"),
                                        "chunk_conf": st.session_state.get('chunk_conf', {})
                                    },
                                    model=model['model']
                                )
                                i+=1
                        elif vector_source == "vector search(gcp)":
                            st.write(f"**Using vector search - remote**")
                            i = 1
                            folder_name = f"{int(time.time())}_id_{uuid.uuid4()}"
                            for item, embedding in all_embeddings:
                                unique_id = str(uuid.uuid4())
                                single_data = {
                                    "id": unique_id, 
                                    "embedding": [float(x) for x in embedding], 
                                    "embedding_metadata": {
                                        "title": item.get("filename"),
                                        "source": "from streamlit dev spike",
                                        "text": item["text"],
                                        "filename": item.get("filename"),
                                        "file_chunk_count": item.get("file_chunk_count"),
                                        "chunk_index": item.get("chunk_index"),
                                        "chunk_conf": st.session_state.get('chunk_conf', {})
                                    }
                                }
                                single_data_json = json.dumps(single_data)
                                st.write(f"saving {i}/{total_chunks} embedding ({item['filename']} #{item['chunk_index']+1}/{item['file_chunk_count']})")
                                upload_string_to_bucket(
                                    bucket_name=model["bucket"],
                                    content=single_data_json,
                                    destination_blob_name=f"{model['model'].replace('/', '-')}_time_{folder_name}/data_{unique_id}.json"
                                )
                                i += 1

                    time.sleep(0.5)

                st.success("✅ Completed!")

        # Individual model buttons (per-model) — updated to use chunk_items + batching
        for i, model in enumerate(models):
            st.markdown(f"**{model['model']}**")
            if st.button(f"Embed, Upload and Index ({model['model']})", key=f"embed_{i}"):
                if total_chunks == 0:
                    st.warning("Chunk the text first!")
                else:
                    st.success(f"Started for {model['model']}.")
                    with st.spinner(text="In progress...", show_time=True, width="content"):
                        st.write(f"(1/5) Loading model: {model['model']}")
                        embed_model = get_model(model['model'])
                        st.write(f"(2/5) Embedding {total_chunks} chunks (batch size {batch_size})")

                        all_embeddings = []
                        processed = 0
                        for batch in batched(chunk_items, batch_size):
                            texts = [item["text"] for item in batch]
                            st.write(f"Embedding batch: {processed+1} -> {processed+len(texts)} / {total_chunks}")
                            embeddings = embed_model.encode(texts)
                            for item, emb in zip(batch, embeddings):
                                all_embeddings.append((item, emb))
                            processed += len(texts)

                        st.write(f"(3/5) Removing model from memory")
                        del embed_model
                        st.write(f"(4/5) Vector to bucket")

                        vector_source = st.session_state.get("source")
                        assert vector_source
                        if vector_source == "sqlite(local)":
                            st.write(f"**Using sqlite - Local**")
                            j = 1
                            for item, embedding in all_embeddings:
                                unique_id = str(uuid.uuid4())
                                st.write(f"saving {j}/{total_chunks} embedding ({item['filename']} #{item['chunk_index']+1}/{item['file_chunk_count']})")
                                insert_doc(
                                    doc_id=unique_id,
                                    content=item["text"],
                                    vec=embedding,
                                    meta={
                                        "title": item.get("filename"),
                                        "source": "from streamlit dev spike",
                                        "text": item["text"],
                                        "filename": item.get("filename"),
                                        "file_chunk_count": item.get("file_chunk_count"),
                                        "chunk_index": item.get("chunk_index"),
                                        "chunk_conf": st.session_state.get('chunk_conf', {})
                                    },
                                    model=model['model']
                                )
                                j+=1
                        elif vector_source == "vector search(gcp)":
                            st.write(f"**Using vector search - remote**")
                            j = 1
                            folder_name = f"{int(time.time())}_id_{uuid.uuid4()}"
                            for item, embedding in all_embeddings:
                                unique_id = str(uuid.uuid4())
                                single_data = {
                                    "id": unique_id,
                                    "embedding": [float(x) for x in embedding],
                                    "embedding_metadata": {
                                        "title": item.get("filename"),
                                        "source": "from streamlit dev spike",
                                        "text": item["text"],
                                        "filename": item.get("filename"),
                                        "file_chunk_count": item.get("file_chunk_count"),
                                        "chunk_index": item.get("chunk_index"),
                                        "chunk_conf": st.session_state.get('chunk_conf', {})
                                    }
                                }
                                single_data_json = json.dumps(single_data)
                                st.write(f"saving {j}/{total_chunks} embedding ({item['filename']} #{item['chunk_index']+1}/{item['file_chunk_count']})")
                                upload_string_to_bucket(
                                    bucket_name=model["bucket"],
                                    content=single_data_json,
                                    destination_blob_name=f"{model['model'].replace('/', '-')}_time_{folder_name}/data_{unique_id}.json"
                                )
                                j += 1

                        st.write(f"(5/5) Indexing")
                        time.sleep(0.5)

                    st.success("✅ Completed!")
# --- END REPLACEMENT ---




