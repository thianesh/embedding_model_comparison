import time
from embedding_models.all_embedding_models import get_model
from sqlite.sample_query import query_knn
import streamlit as st
import pandas as pd
import textwrap

vector_source = st.session_state.get("source") 
assert vector_source

st.write(f"### Query Page: {vector_source}")
models = st.session_state.get("table_data", [])

query_text = st.text_area("Enter your query")
if st.button("Search in all models."):
    if not query_text:
        st.warning("Please submit a valid query.")
    else:
        st.success(f"Searching Semantically.")
        
        with st.spinner(text="In progress...", show_time=True, width="content"):
            # for i in range(5):
            #     time.sleep(0.5)  # simulate work
            #     st.write(f"Step {i+1}/5 done")  # dynamic message updates
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
            # all_model_result

            # Build a rectangular table where columns = model names and each cell shows: score, id, truncated text
            models = list(all_model_result.keys())
            max_rows = max((len(lst) for lst in all_model_result.values()), default=0)

            table_rows = []
            for row_idx in range(max_rows):
                row = {}
                for m in models:
                    try:
                        entry = all_model_result[m][row_idx]
                        score = entry[0]
                        uid = entry[1]
                        # text may be at position 2 or inside meta - user said "text content" and provided it at pos 2
                        text_content = entry[2] if len(entry) > 2 else ""
                        # make readable, single-line and truncated
                        single = " ".join(text_content.split())
                        short = textwrap.shorten(single, width=220, placeholder="...")
                        # format: score (4dp), id, short text (keeps things compact)
                        cell = f"{score:.4f}\nID: {uid}\n{short}"
                    except Exception:
                        cell = ""
                    row[m] = cell
                table_rows.append(row)

            # Convert to DataFrame and display
            if table_rows:
                df = pd.DataFrame(table_rows)
                st.markdown("### Search Results (columns = models; each cell: score, id, truncated text)")
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No search results to show.")

            # Additionally: show full entries per model in expanders (so user can read full text + metadata)
            st.markdown("---")
            st.markdown("### Full results by model")
            for m in models:
                with st.expander(m, expanded=False):
                    entries = all_model_result.get(m, [])
                    for idx, e in enumerate(entries):
                        score = e[0] if len(e) > 0 else None
                        uid = e[1] if len(e) > 1 else None
                        text_content = e[2] if len(e) > 2 else ""
                        meta = e[3] if len(e) > 3 else None
                        st.markdown(f"**Result #{idx+1}** — **score:** {score:.4f}   |   `id:` `{uid}`")
                        st.write(text_content)
                        if meta:
                            st.caption(f"meta: {meta}")
                        st.markdown("---")

if not models:
    st.info("No models found. Add some models on the Models page.")
else:
    for i, model in enumerate(models):
        if st.button(f"Search in ({model['model']})", key=f"search_{i}"):
            if not query_text:
                st.warning("Please submit a valid query.")
            else:
                st.success(f"Searching in ({model['model']}) Semantically.")
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

                    # all_model_result

                    # Build a rectangular table where columns = model names and each cell shows: score, id, truncated text
                    models = list(all_model_result.keys())
                    max_rows = max((len(lst) for lst in all_model_result.values()), default=0)

                    table_rows = []
                    for row_idx in range(max_rows):
                        row = {}
                        for m in models:
                            try:
                                entry = all_model_result[m][row_idx]
                                score = entry[0]
                                uid = entry[1]
                                # text may be at position 2 or inside meta - user said "text content" and provided it at pos 2
                                text_content = entry[2] if len(entry) > 2 else ""
                                # make readable, single-line and truncated
                                single = " ".join(text_content.split())
                                short = textwrap.shorten(single, width=220, placeholder="...")
                                # format: score (4dp), id, short text (keeps things compact)
                                cell = f"{score:.4f}\nID: {uid}\n{short}"
                            except Exception:
                                cell = ""
                            row[m] = cell
                        table_rows.append(row)

                    # Convert to DataFrame and display
                    if table_rows:
                        df = pd.DataFrame(table_rows)
                        st.markdown("### Search Results (columns = models; each cell: score, id, truncated text)")
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No search results to show.")

                    # Additionally: show full entries per model in expanders (so user can read full text + metadata)
                    st.markdown("---")
                    st.markdown("### Full results by model")
                    for m in models:
                        with st.expander(m, expanded=False):
                            entries = all_model_result.get(m, [])
                            for idx, e in enumerate(entries):
                                score = e[0] if len(e) > 0 else None
                                uid = e[1] if len(e) > 1 else None
                                text_content = e[2] if len(e) > 2 else ""
                                meta = e[3] if len(e) > 3 else None
                                st.markdown(f"**Result #{idx+1}** — **score:** {score:.4f}   |   `id:` `{uid}`")
                                st.write(text_content)
                                if meta:
                                    st.caption(f"meta: {meta}")
                                st.markdown("---")


st.write("Current table data:", st.session_state.table_data)