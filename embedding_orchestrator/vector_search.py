from typing import List
from google.cloud import aiplatform
import os
from dotenv import load_dotenv

# from ..sample.sample_embedding import embedding

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample.sample_embedding import embedding


load_dotenv()

VECTOR_INDEX_PREFIX = os.getenv("VECTOR_INDEX_PREFIX")

# To add index to the vector search,
# first upload jsonl to the bucket,
# sample json {"id": "vector_001", "vector": [0.12, 0.45, 0.33, ..., 0.78], "metadata": {"title": "COVID-19 Study", "source": "pubmed"}}
# then call the import_index_data

aiplatform.init(project='talk-to-your-records', location='us-central1')

def vector_search_find_neighbors(
    index_endpoint_name: str,
    deployed_index_id: str,
    queries: List[List[float]],
    num_neighbors: int,
) -> List[
    List[aiplatform.matching_engine.matching_engine_index_endpoint.MatchNeighbor]
]:
    """Query the vector search index.

    Args:
        project (str): Required. Project ID
        location (str): Required. The region name
        index_endpoint_name (str): Required. Index endpoint to run the query
        against.
        deployed_index_id (str): Required. The ID of the DeployedIndex to run
        the queries against.
        queries (List[List[float]]): Required. A list of queries. Each query is
        a list of floats, representing a single embedding.
        num_neighbors (int): Required. The number of neighbors to return.

    Returns:
        List[List[aiplatform.matching_engine.matching_engine_index_endpoint.MatchNeighbor]] - A list of nearest neighbors for each query.
    """
    # Create the index endpoint instance from an existing endpoint.
    my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=index_endpoint_name
    )

    # Query the index endpoint for the nearest neighbors.
    return my_index_endpoint.find_neighbors(
        deployed_index_id=deployed_index_id,
        queries=queries,
        num_neighbors=num_neighbors,
        return_full_datapoint=True,
    )



# deployed_index_id = "id_test_vector_search_undeploy_this_if_seen_up",
# results = vector_search_find_neighbors(
#     index_endpoint_name = "3063065672146747392",
#     deployed_index_id = "id_test_vector_search_undeploy_this_if_seen_up",
#     queries= [embedding],
#     num_neighbors = 5,
# )

# # print(results)
# for neighbor_list in results:
#     for neighbor in neighbor_list:
#         # Safely access the dictionary inside the MatchNeighbor object
#         metadata = getattr(neighbor, 'embedding_metadata', None)

#         if metadata:
#             print(metadata.get('text'))
#             # You can also use: print(neighbor.embedding_metadata.get('text'))
#         else:
#             print("Metadata field is missing or empty.")