models_to_use =  [
        {"model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", "bucket": "vector-db-storage-exp", "index": "5999368648727199744"},
        {"model": "Qwen/Qwen3-Embedding-0.6B", "bucket": "my-bucket-2", "index": "bio-index"},
        {"model": "abhinand/MedEmbed-small-v0.1", "bucket": "my-bucket-3", "index": "st-index"},
        {"model": "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1", "bucket": "my-bucket-4", "index": "scibert-index"},
        {"model": "google/embeddinggemma-300m", "bucket": "my-bucket-4", "index": "scibert-index"},
        {"model": "BAAI/bge-base-en-v1.5", "bucket": "my-bucket-4", "index": "scibert-index"}
    ]
# remote code
# {"model": "Snowflake/snowflake-arctic-embed-m-v2.0", "bucket": "my-bucket-4", "index": "scibert-index"},
# {"model": "Alibaba-NLP/gte-multilingual-base", "bucket": "my-bucket-4", "index": "scibert-index"}

# Sample runs and dimensions
'''
Shape for model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract 
(2, 768)
Shape for model: Qwen/Qwen3-Embedding-0.6B 
(2, 1024)
Shape for model: abhinand/MedEmbed-small-v0.1 
(2, 384)
Shape for model: HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1 
(2, 896)
Shape for model: google/embeddinggemma-300m 
(2, 768)
Shape for model: BAAI/bge-base-en-v1.5 
(2, 768)
'''