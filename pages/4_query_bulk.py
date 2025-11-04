# search_page.py  (replace your existing page script with this)
import time
import json
import textwrap
from typing import List, Optional, Tuple, Dict, Any
import streamlit as st
import pandas as pd

from embedding_models.all_embedding_models import get_model
from sqlite.sample_query import query_knn

from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()   
import numpy as np

# Initialize client
client = OpenAI()

# ---------- Helpers ----------
def _normalize_text(s: Optional[str]) -> str:
    """Normalize for comparison: strip, collapse whitespace, lower-case."""
    if s is None:
        return ""
    return " ".join(str(s).split()).strip().lower()

def _check_ground_in_text(gt_norm: str, text: Optional[str]) -> bool:
    """Return True if normalized ground truth is a substring of normalized text."""
    if not gt_norm:
        return False
    return gt_norm in _normalize_text(text)

def parse_json_list_from_bytes(b: bytes) -> Optional[List[str]]:
    """Parse bytes from uploaded file into list of strings (or list of dict->text)."""
    try:
        parsed = json.loads(b)
    except Exception:
        # try decode then load (rare)
        try:
            parsed = json.loads(b.decode("utf-8"))
        except Exception as e:
            st.error(f"Could not parse uploaded JSON: {e}")
            return None

    return _normalize_or_extract_list(parsed)

def parse_json_list_from_text(s: str) -> Optional[List[str]]:
    """Parse a pasted text blob into list. Accepts:
       - JSON array (top-level)
       - newline separated plain lines
       - newline separated JSON objects (one per line)
    """
    s = s.strip()
    if not s:
        return None
    # Try JSON parse
    try:
        parsed = json.loads(s)
        return _normalize_or_extract_list(parsed)
    except Exception:
        # try newline-separated JSON / plain lines
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        out = []
        for ln in lines:
            # try each line as JSON
            try:
                obj = json.loads(ln)
                if isinstance(obj, str):
                    out.append(obj)
                elif isinstance(obj, dict):
                    # try common keys
                    text_val = obj.get("text") or obj.get("query") or obj.get("chunk") or obj.get("content") or ""
                    out.append(text_val)
                else:
                    out.append(str(obj))
            except Exception:
                # fallback: use the raw line as a string
                out.append(ln)
        return out if out else None

def _normalize_or_extract_list(parsed) -> Optional[List[str]]:
    """Turn parsed JSON into list[str] extracting common keys if needed."""
    if isinstance(parsed, list):
        out = []
        for item in parsed:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                # prefer 'query' or 'text' or 'chunk' or 'content'
                text_val = item.get("query") or item.get("text") or item.get("chunk") or item.get("content") or ""
                out.append(text_val)
            else:
                out.append(str(item))
        return out
    elif isinstance(parsed, dict):
        # support single object with list inside e.g. {"items": [...]} or paired [{query,reference},...]
        # if dict has keys 'queries' or 'items' that are lists, extract them
        for key in ("queries", "items", "data"):
            if key in parsed and isinstance(parsed[key], list):
                return _normalize_or_extract_list(parsed[key])
        # maybe it's a single dict representing one pair; return its query if present
        if "query" in parsed:
            return [parsed.get("query", "")]
        st.error("Uploaded JSON is a dict, expected a top-level array or newline-separated items.")
        return None
    else:
        st.error("Uploaded content is not a JSON array.")
        return None

def load_queries_and_refs_from_paired_text(s: str) -> Optional[Tuple[List[str], List[str]]]:
    """If user pasted a single JSON containing paired objects like [{"query": "...", "reference":"..."} ...]"""
    try:
        parsed = json.loads(s)
    except Exception:
        return None
    if not isinstance(parsed, list):
        return None
    queries = []
    refs = []
    for item in parsed:
        if isinstance(item, dict):
            q = item.get("query") or item.get("text") or item.get("question") or ""
            r = item.get("reference") or item.get("ref") or item.get("ground_truth") or item.get("chunk") or item.get("content") or ""
            queries.append(q)
            refs.append(r)
        else:
            # not paired objects
            return None
    return queries, refs

def build_rows_for_single_query(
    query_text: str,
    reference_chunk: str,
    all_model_result: Dict[str, List],
    k: int,
    truncate_width: int = 220
) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[int]]]:
    """
    Given query_text and reference_chunk and the dictionary all_model_result (model -> list of entries),
    return rows that include columns: model, query, reference_chunk, rank, score, retrieved_chunk, ground_truth_match, ground_truth_rank.
    Also returns per_model_rank dict (first matched rank per model).
    """
    rows = []
    models_list = list(all_model_result.keys())
    per_model_rank = {m: None for m in models_list}
    gt_norm = _normalize_text(reference_chunk)

    # compute per-model rank
    if gt_norm:
        for m in models_list:
            entries = all_model_result.get(m, [])
            for idx, e in enumerate(entries):
                text_content = ""
                if len(e) > 2:
                    text_content = e[2] or ""
                elif len(e) > 3 and isinstance(e[3], dict):
                    meta = e[3]
                    text_content = meta.get("text", "") or meta.get("content", "") or meta.get("chunk", "") or ""
                if _check_ground_in_text(gt_norm, text_content):
                    per_model_rank[m] = idx + 1
                    break

    for m in models_list:
        entries = all_model_result.get(m, [])
        for idx in range(min(len(entries), k)):
            e = entries[idx]
            score = e[0] if len(e) > 0 else None
            uid = e[1] if len(e) > 1 else None
            text_content = e[2] if len(e) > 2 else ""
            if not text_content and len(e) > 3 and isinstance(e[3], dict):
                meta = e[3]
                text_content = meta.get("text", "") or meta.get("content", "") or meta.get("chunk", "") or ""
            single = " ".join(str(text_content).split())
            short = textwrap.shorten(single, width=truncate_width, placeholder="...")
            gt_match = _check_ground_in_text(gt_norm, text_content)
            gt_rank = per_model_rank.get(m) or ""
            rows.append({
                "model": m,
                "query": query_text,
                "reference_chunk": reference_chunk,
                "rank": idx + 1,
                "score": f"{score:.4f}" if (score is not None) else "",
                "uid": uid,
                "retrieved_chunk": short,
                "ground_truth_match": bool(gt_match),
                "ground_truth_rank": gt_rank if gt_match else per_model_rank.get(m) or ""
            })
    return rows, per_model_rank

# ---------- UI & Inputs ----------
vector_source = st.session_state.get("source")
if not vector_source:
    # Provide clearer message instead of assert to avoid breaking UI
    st.error("Vector source missing in session_state. Please set session_state['source'].")
    st.stop()

st.title(f"Bulk Query / Evaluation Page: {vector_source}")

models = st.session_state.get("table_data", [])
if not models:
    st.info("No models found. Add some models on the Models page.")
    st.stop()

st.markdown(
    "You can either **paste JSON** (in the text boxes) or **upload JSON files**. "
    "Supported formats:\n\n"
    "- `queries.json`: top-level array of strings, or newline-separated strings\n"
    "- `ground_truths.json`: top-level array of strings or objects with `text`/`chunk`\n"
    "- **Paired single JSON**: `[{'query':'...','reference':'...'}, ...]` — paste/upload into either box and it will be detected."
)

# Paste inputs
pasted_paired = st.text_area("Paste paired JSON (optional) — array of objects with `query` and `reference` keys. If provided, this will be used preferentially.", height=140)
st.markdown("Or paste separate JSON lists below (used if paired JSON absent).")
pasted_queries = st.text_area("Paste queries JSON (array or newline-separated)", height=140)
pasted_gts = st.text_area("Paste ground-truth JSON (array or newline-separated)", height=140)

# Upload inputs
uploaded_queries = st.file_uploader("Or upload queries JSON file", type=["json"])
uploaded_gts = st.file_uploader("Or upload ground-truth JSON file", type=["json"])

k = st.number_input("Top K to retrieve per query (per model)", value=5, min_value=1, max_value=200, step=1)
truncate_width = st.number_input("Truncate width for displayed chunk text", value=220, min_value=50, max_value=4000, step=10)

# ---------- Parse inputs ----------
queries_list: Optional[List[str]] = None
gts_list: Optional[List[str]] = None

# If paired pasted JSON is present, try to use it
if pasted_paired and pasted_paired.strip():
    paired = load_queries_and_refs_from_paired_text(pasted_paired)
    if paired:
        queries_list, gts_list = paired
    else:
        st.warning("Paired pasted JSON not recognized as array of objects with 'query' & 'reference' keys. Will fallback to other inputs.")

# If not paired, parse separate pasted fields first (pasted takes precedence over uploaded)
if queries_list is None and pasted_queries and pasted_queries.strip():
    parsed = None
    # try detect single paired list inside pasted_queries (rare)
    paired_try = load_queries_and_refs_from_paired_text(pasted_queries)
    if paired_try:
        queries_list, gts_list = paired_try
    else:
        parsed = parse_json_list_from_text(pasted_queries)
        if parsed:
            queries_list = parsed

if gts_list is None and pasted_gts and pasted_gts.strip():
    parsed = parse_json_list_from_text(pasted_gts)
    if parsed:
        gts_list = parsed

# If still missing, try uploaded files
if queries_list is None and uploaded_queries is not None:
    try:
        b = uploaded_queries.read()
        queries_list = parse_json_list_from_bytes(b)
    except Exception as e:
        st.error(f"Failed to read uploaded queries file: {e}")

if gts_list is None and uploaded_gts is not None:
    try:
        b = uploaded_gts.read()
        gts_list = parse_json_list_from_bytes(b)
    except Exception as e:
        st.error(f"Failed to read uploaded ground-truth file: {e}")

# ---------- Run bulk search ----------
if st.button("Run bulk search for provided queries"):
    # Validate queries
    st.write(f"queries_list length: {len(queries_list) if queries_list else 0}")
    st.write(f"gts_list length: {len(gts_list) if gts_list else 0}")

     # Must have queries
    if not queries_list:
        st.warning("No queries provided. Paste or upload a queries JSON.")
    else:
        # If ground truth missing, continue but warn
        if not gts_list:
            st.warning("No ground-truth list provided — proceeding without references. 'reference_chunk' column will be empty.")
        else:
            if len(queries_list) != len(gts_list):
                st.warning(
                    f"queries length = {len(queries_list)}, ground-truth length = {len(gts_list)}. "
                    "Processing up to the shorter length (min). If you intended 1:1 mapping, upload matching lists or a paired JSON."
                )

        n = min(len(queries_list), len(gts_list)) if gts_list else len(queries_list)
        st.info(f"Processing {n} queries across {len(models)} models (top {k} per model).")

        overall_rows: List[Dict[str, Any]] = []
        overall_per_query_per_model_rank: Dict[str, Dict[str, Optional[int]]] = {}

        # Preload and cache models to avoid repeated loads:
        model_cache: Dict[str, Any] = {}
        try:
            for m in models:
                mname = m["model"]
                st.write(f"Loading model {mname} into cache...")
                try:
                    model_obj = get_model(mname)
                    model_cache[mname] = model_obj
                except Exception as e:
                    st.error(f"Failed to load model {mname}: {e}")
                    # Do not stop; leave it out of cache (will be skipped)
                    model_cache[mname] = None

            # For each query, run searches across models
            for qi in range(n):
                q_text = queries_list[qi]
                ref_chunk = gts_list[qi] if gts_list else ""
                st.write(f"---\n### Query {qi+1}/{n}")
                st.write(f"> {q_text}")
                if ref_chunk:
                    st.write("**Reference chunk provided.**")
                else:
                    st.write("*No reference chunk for this query.*")

                # For each model, embed and query. We'll embed per-model in batch mode for this single query set,
                # but since we process queries sequentially we embed one query at a time. To speed up many queries,
                # consider batching all queries per model (not implemented here to preserve memory).
                all_model_result: Dict[str, List] = {}
                for idx, model in enumerate(models):
                    mname = model["model"]
                    st.write(f"(Model {idx+1}/{len(models)}) {mname}")
                    embed_model = model_cache.get(mname)
                    if embed_model is None:
                        st.warning(f"Model {mname} not available (failed to load earlier). Skipping.")
                        continue

                    # Try batch encode (single item list), most encoders support list input
                    try:
                        embeddings = None
                        if type(embed_model) is str:
                            st.write("Using OpenAI API for embeddings")
                            response = client.embeddings.create(
                                model=embed_model.replace("openai/", ""),   # or "text-embedding-3-large"
                                input=[q_text]
                            )
                            def openai_embedding_to_numpy(response):
                                """
                                Convert OpenAI embedding response into a 2D NumPy array.
                                Shape -> (total_embeddings, embedding_vector_size)
                                """
                                # Extract embeddings from response.data
                                embeddings = [item.embedding for item in response.data]
                                return np.array(embeddings, dtype=np.float32)
                            
                            emb_vec = openai_embedding_to_numpy(response)[0]
                            # emb_vec = embeddings[0] if embeddings is not None and len(embeddings) > 0 else None
                            emb_vec.shape
                            # emb_vec
                        else:
                            embeddings = embed_model.encode([q_text])
                            emb_vec = embeddings[0] if embeddings is not None and len(embeddings) > 0 else None
                            emb_vec.shape
                    except Exception as e:
                        st.warning(f"Batch encode failed for model {mname}, trying single encode: {e}")
                        try:
                            emb_vec = embed_model.encode(q_text)
                        except Exception as e2:
                            st.error(f"Failed to embed query with model {mname}: {e2}")
                            continue

                    # Query vector DB
                    try:
                        query_result = query_knn(
                            query_vec=emb_vec,
                            k=k,
                            model=mname
                        )
                        # Expect list of [score, uid, text, meta?]
                        if not isinstance(query_result, list):
                            st.warning(f"query_knn returned unexpected type for model {mname}. Expected list, got {type(query_result)}")
                            query_result = []
                    except Exception as e:
                        st.error(f"query_knn failed for model {mname}: {e}")
                        query_result = []

                    all_model_result[mname] = query_result
                    time.sleep(0.05)

                # Build rows for this query
                rows_for_q, per_model_rank = build_rows_for_single_query(
                    query_text=q_text,
                    reference_chunk=ref_chunk or "",
                    all_model_result=all_model_result,
                    k=k,
                    truncate_width=truncate_width
                )
                overall_rows.extend(rows_for_q)
                overall_per_query_per_model_rank[q_text] = per_model_rank

            # Done processing queries
            if overall_rows:
                df = pd.DataFrame(overall_rows)
                col_order = ["model", "query", "reference_chunk", "rank", "score", "uid", "retrieved_chunk", "ground_truth_match", "ground_truth_rank"]
                cols_present = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
                df = df[cols_present]
                st.markdown("### Bulk Search Results")
                st.dataframe(df, use_container_width=True)

                st.markdown("---")
                st.markdown("### Per-query summary (Ground-truth found? first rank per model)")
                for q_text, per_model_rank in overall_per_query_per_model_rank.items():
                    with st.expander(q_text, expanded=False):
                        summary_rows = []
                        for m, r in per_model_rank.items():
                            summary_rows.append({"model": m, "first_matched_rank": r or ""})
                        summary_df = pd.DataFrame(summary_rows)
                        st.table(summary_df)

                csv = df.to_csv(index=False)
                st.download_button("Download results as CSV", data=csv, file_name="bulk_search_results.csv", mime="text/csv")
            else:
                st.info("No results produced.")
        
            all_model_result
        finally:
            # Clean up models from memory
            for mname, obj in model_cache.items():
                try:
                    if obj is not None:
                        del obj
                except Exception:
                    pass

st.write("Current table data (models):", st.session_state.get("table_data"))
