import time
from embedding_models.all_embedding_models import get_model
import streamlit as st

st.write("### Query Page")
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
            for index, model in enumerate(models):
                st.write(f"#### {model['model']} {index + 1}/{len(models)}")
                st.write(f"(1/5) Loading model: {model['model']}")
                embed_model = get_model(model['model'])
                st.write(f"(2/5) Embedding query")
                embeddings = embed_model.encode(query_text)
                embeddings
                st.write(f"(3/5) Removing model from memory")
                del embed_model
                st.write(f"(4/5) Dense Vector Search")
                time.sleep(0.5)
                st.write(f"(5/5) showing result")

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
                    for i in range(5):
                        time.sleep(0.5)  # simulate work
                        st.write(f"Step {i+1}/5 done")  # dynamic message updates


st.write("Current table data:", st.session_state.table_data)