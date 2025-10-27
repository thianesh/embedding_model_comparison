# search_page.py  (replace your existing page script with this)
import time
import textwrap
from embedding_models.all_embedding_models import get_model
from sqlite.sample_query import query_knn
import streamlit as st
import pandas as pd

vector_source = st.session_state.get("source")
assert vector_source

st.write(f"### Query Page: {vector_source}")
models = st.session_state.get("table_data", [])

# --- Inputs ---
query_text = st.text_area("Enter your query")
ground_truth_chunk = st.text_area("Ground truth chunk (optional) - paste the exact chunk you expect to be retrieved")

def _normalize_text(s: str) -> str:
    """Normalize for comparison: strip, collapse whitespace, lower-case."""
    if s is None:
        return ""
    return " ".join(str(s).split()).strip().lower()

ground_norm = _normalize_text(ground_truth_chunk)

def _check_ground_in_text(gt_norm, text) -> bool:
    """Return True if normalized ground truth is a substring of normalized text."""
    if not gt_norm:
        return False
    return gt_norm in _normalize_text(text)

# --- Helper to build table rows given all_model_result (dict: model -> list-of-entries) ---
def build_split_table(all_model_result: dict, truncate_width: int = 220):
    # Prepare list of models in deterministic order
    models_list = list(all_model_result.keys())
    max_rows = max((len(lst) for lst in all_model_result.values()), default=0)

    table_rows = []
    # We'll also keep per-model ground-truth rank info (first matched index or None)
    per_model_rank = {m: None for m in models_list}

    # Precompute ranks per model (so ground_truth_rank column is filled only on matched row)
    if ground_norm:
        for m in models_list:
            entries = all_model_result.get(m, [])
            for idx, e in enumerate(entries):
                text_content = e[2] if len(e) > 2 else ""
                # if no text at pos 2, check meta too
                if not text_content and len(e) > 3:
                    meta = e[3]
                    # if meta is dict and has 'text' key
                    if isinstance(meta, dict):
                        text_content = meta.get("text", "") or meta.get("content", "") or ""
                if _check_ground_in_text(ground_norm, text_content):
                    per_model_rank[m] = idx + 1
                    break

    for row_idx in range(max_rows):
        row = {}
        for m in models_list:
            try:
                entry = all_model_result[m][row_idx]
                score = entry[0] if len(entry) > 0 else None
                uid = entry[1] if len(entry) > 1 else None
                text_content = entry[2] if len(entry) > 2 else ""
                if not text_content and len(entry) > 3 and isinstance(entry[3], dict):
                    # try common meta keys if text is not in pos 2
                    meta = entry[3]
                    text_content = meta.get("text", "") or meta.get("content", "") or meta.get("chunk", "") or ""
                # readable single-line and truncated for table display
                single = " ".join(str(text_content).split())
                short = textwrap.shorten(single, width=truncate_width, placeholder="...")
                # determine ground truth match for this row
                gt_match = _check_ground_in_text(ground_norm, text_content)
                gt_rank = per_model_rank.get(m) if per_model_rank.get(m) is not None else ""
                # Populate four split columns per your spec
                row[f"{m} | similarity_score"] = f"{score:.4f}" if (score is not None) else ""
                row[f"{m} | ground_truth_match"] = "True" if gt_match else "False"
                # only show rank on the row where it matches (so other rows blank)
                row[f"{m} | ground_truth_rank"] = str(gt_rank) if (gt_match and gt_rank) else (str(gt_rank) if (gt_rank and gt_rank == row_idx+1) else (str(gt_rank) if gt_rank and row_idx==gt_rank-1 else ""))
                row[f"{m} | retrieved_chunk"] = short
            except Exception:
                # empty cells for missing rows
                row[f"{m} | similarity_score"] = ""
                row[f"{m} | ground_truth_match"] = ""
                row[f"{m} | ground_truth_rank"] = ""
                row[f"{m} | retrieved_chunk"] = ""
        table_rows.append(row)
    return table_rows, per_model_rank

# --- Multi-model search button ---
if st.button("Search in all models."):
    if not query_text:
        st.warning("Please submit a valid query.")
    else:
        st.success("Searching semantically.")
        with st.spinner(text="In progress...", show_time=True, width="content"):
            all_model_result = {}
            for index, model in enumerate(models):
                st.write(f"#### {model['model']} {index + 1}/{len(models)}")
                st.write(f"(1/5) Loading model: {model['model']}")
                embed_model = get_model(model['model'])
                st.write(f"(2/5) Embedding query")
                embeddings = embed_model.encode(query_text)
                st.write(f"(3/5) Removing model from memory")
                del embed_model
                st.write(f"(4/5) Dense Vector Search")
                query_result = query_knn(
                    query_vec=embeddings,
                    k=5,
                    model=model['model']
                )
                all_model_result[model['model']] = query_result
                time.sleep(0.5)
                st.write(f"(5/5) showing result")

            # Build split table
            table_rows, per_model_rank = build_split_table(all_model_result)
            if table_rows:
                df = pd.DataFrame(table_rows)
                st.markdown("### Search Results (split columns per model)")
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No search results to show.")

            # Full entries per model with ground-truth indicators
            st.markdown("---")
            st.markdown("### Full results by model")
            for m in all_model_result.keys():
                with st.expander(m, expanded=False):
                    entries = all_model_result.get(m, [])
                    gt_rank = per_model_rank.get(m)
                    st.write(f"Ground truth present: {'True' if gt_rank else 'False'}")
                    if gt_rank:
                        st.write(f"Ground truth rank: {gt_rank}")
                    for idx, e in enumerate(entries):
                        score = e[0] if len(e) > 0 else None
                        uid = e[1] if len(e) > 1 else None
                        text_content = e[2] if len(e) > 2 else ""
                        meta = e[3] if len(e) > 3 else None
                        gt_match_row = _check_ground_in_text(ground_norm, text_content)
                        st.markdown(f"**Result #{idx+1}** — **score:** {score:.4f}   |   `id:` `{uid}`   | ground_truth_match: {gt_match_row}")
                        st.write(text_content)
                        if meta:
                            st.caption(f"meta: {meta}")
                        st.markdown("---")

# --- Per-model search buttons (individual) ---
if not models:
    st.info("No models found. Add some models on the Models page.")
else:
    for i, model in enumerate(models):
        if st.button(f"Search in ({model['model']})", key=f"search_{i}"):
            if not query_text:
                st.warning("Please submit a valid query.")
            else:
                st.success(f"Searching in ({model['model']}) semantically.")
                with st.spinner(text="In progress...", show_time=True, width="content"):
                    all_model_result = {}
                    st.write(f"#### {model['model']}")
                    st.write(f"(1/5) Loading model: {model['model']}")
                    embed_model = get_model(model['model'])
                    st.write(f"(2/5) Embedding query")
                    embeddings = embed_model.encode(query_text)
                    st.write(f"(3/5) Removing model from memory")
                    del embed_model
                    st.write(f"(4/5) Dense Vector Search")
                    query_result = query_knn(
                        query_vec=embeddings,
                        k=5,
                        model=model['model']
                    )
                    all_model_result[model['model']] = query_result
                    time.sleep(0.5)
                    st.write(f"(5/5) showing result")

                    # Build split table (single-model)
                    table_rows, per_model_rank = build_split_table(all_model_result)
                    if table_rows:
                        df = pd.DataFrame(table_rows)
                        st.markdown("### Search Results (split columns per model)")
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No search results to show.")

                    # Full entries per model
                    st.markdown("---")
                    st.markdown("### Full results by model")
                    for m in all_model_result.keys():
                        with st.expander(m, expanded=False):
                            entries = all_model_result.get(m, [])
                            gt_rank = per_model_rank.get(m)
                            st.write(f"Ground truth present: {'True' if gt_rank else 'False'}")
                            if gt_rank:
                                st.write(f"Ground truth rank: {gt_rank}")
                            for idx, e in enumerate(entries):
                                score = e[0] if len(e) > 0 else None
                                uid = e[1] if len(e) > 1 else None
                                text_content = e[2] if len(e) > 2 else ""
                                meta = e[3] if len(e) > 3 else None
                                gt_match_row = _check_ground_in_text(ground_norm, text_content)
                                st.markdown(f"**Result #{idx+1}** — **score:** {score:.4f}   |   `id:` `{uid}`   | ground_truth_match: {gt_match_row}")
                                st.write(text_content)
                                if meta:
                                    st.caption(f"meta: {meta}")
                                st.markdown("---")

st.write("Current table data:", st.session_state.table_data)
