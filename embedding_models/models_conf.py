        # {"model": "infgrad/jasper_en_vision_language_v1", "bucket": "", "index": ""}, # 1B
        # {"model": "jinaai/jina-embeddings-v3", "bucket": "", "index": ""}, # 572 M

models_to_use =  [
        # {"model": "NeuML/pubmedbert-base-embeddings", "bucket": "vector-db-storage-exp", "index": "5999368648727199744"},

        # {"model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "bucket": "", "index": ""}, # just 1B but 4th in rankings 

        # {"model": "intfloat/multilingual-e5-large-instruct", "bucket": "", "index": ""}, # 560 M
        # {"model": "Snowflake/snowflake-arctic-embed-l-v2.0", "bucket": "", "index": ""}, # 560 M

        # {"model": "Qwen/Qwen3-Embedding-0.6B", "bucket": "my-bucket-2", "index": "bio-index"},
        # {"model": "abhinand/MedEmbed-small-v0.1", "bucket": "my-bucket-3", "index": "st-index"},
        # {"model": "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1", "bucket": "my-bucket-4", "index": "scibert-index"},
        # {"model": "google/embeddinggemma-300m", "bucket": "my-bucket-4", "index": "scibert-index"},
        # {"model": "BAAI/bge-base-en-v1.5", "bucket": "my-bucket-4", "index": "scibert-index"},

        {"model": "openai/text-embedding-3-small", "bucket": "my-bucket-4", "index": "scibert-index"},
        {"model": "openai/text-embedding-3-large", "bucket": "my-bucket-4", "index": "scibert-index"},
        {"model": "openai/text-embedding-ada-002", "bucket": "my-bucket-4", "index": "scibert-index"}
    ]
# remote codes
# {"model": "Snowflake/snowflake-arctic-embed-m-v2.0", "bucket": "my-bucket-4", "index": "scibert-index"},
# {"model": "Alibaba-NLP/gte-multilingual-base", "bucket": "my-bucket-4", "index": "scibert-index"}

# Sample runs and dimensions
'''
Shape for model: NeuML/pubmedbert-base-embeddings 
(2, 768)
Time(Sec) took for embedding in CPU 7950X: 0.01869487762451172 per text

Shape for model: Alibaba-NLP/gte-Qwen2-1.5B-instruct 
(2, 1536)
Time(Sec) took for embedding in CPU 7950X: 0.11034286022186279 per text

Shape for model: intfloat/multilingual-e5-large-instruct 
(2, 1024)
Time(Sec) took for embedding in CPU 7950X: 0.03926730155944824 per text

Shape for model: Snowflake/snowflake-arctic-embed-l-v2.0 
(2, 1024)
Time(Sec) took for embedding in CPU 7950X: 0.03680157661437988 per text

Shape for model: Qwen/Qwen3-Embedding-0.6B 
(2, 1024)
Time(Sec) took for embedding in CPU 7950X: 0.043299198150634766 per text

Shape for model: abhinand/MedEmbed-small-v0.1 
(2, 384)
Time(Sec) took for embedding in CPU 7950X: 0.009679436683654785 per text

Shape for model: HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1 
(2, 896)
Time(Sec) took for embedding in CPU 7950X: 0.03458070755004883 per text

Shape for model: google/embeddinggemma-300m 
(2, 768)
Time(Sec) took for embedding in CPU 7950X: 0.019324660301208496 per text

Shape for model: BAAI/bge-base-en-v1.5 
(2, 768)
Time(Sec) took for embedding in CPU 7950X: 0.02005946636199951 per text
'''