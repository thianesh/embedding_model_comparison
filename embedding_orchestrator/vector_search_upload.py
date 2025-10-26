from google.cloud import aiplatform
import os
from dotenv import load_dotenv

load_dotenv()

VECTOR_INDEX_PREFIX = os.getenv("VECTOR_INDEX_PREFIX")

# To add index to the vector search,
# first upload jsonl to the bucket,
# sample json {"id": "vector_001", "vector": [0.12, 0.45, 0.33, ..., 0.78], "metadata": {"title": "COVID-19 Study", "source": "pubmed"}}
# then call the import_index_data

aiplatform.init(project='talk-to-your-records', location='us-central1')

def index_data(bucket, vector_index):
    # Reference the existing index
    index = aiplatform.MatchingEngineIndex(
        index_name = vector_index
    )

    # index.import_index_data(
    #     gcs_source_uris=[bucket]
    # )

    print(index)

    index.update_embeddings(
        contents_delta_uri = bucket,
        is_complete_overwrite = False
    )

index_data(
    "gs://vector-db-storage-exp/microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract_time_1761405783_id_738d4fc3-a600-4743-a482-76b14c440316",
    "5999368648727199744"
)
