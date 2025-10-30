# token_chunking.py
from transformers import AutoTokenizer
from typing import List, Tuple

def token_level_chunks(
    text: str,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    add_special_tokens: bool = False
) -> List[str]:
    """
    Return a list of text chunks where each chunk corresponds to ~chunk_size tokens
    (tokenized by model_name's tokenizer) with chunk_overlap tokens overlap.

    Uses tokenizer(..., return_offsets_mapping=True) and maps token ranges
    back to original character spans to preserve spaces/punctuation exactly.

    - text: full input text (str)
    - model_name: HF model id to load tokenizer from (e.g. "sentence-transformers/all-MiniLM-L6-v2")
    - chunk_size: tokens per chunk
    - chunk_overlap: number of tokens overlapped between consecutive chunks
    - add_special_tokens: whether to include tokenizer special tokens in counting (default False)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Tokenize and get offsets for char->token mapping
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=add_special_tokens,
        truncation=False,
    )

    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]  # list of (char_start, char_end) for each token

    if not input_ids:
        return []

    stride = chunk_size - chunk_overlap
    if stride <= 0:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    n = len(input_ids)
    start_token = 0
    while start_token < n:
        end_token = min(start_token + chunk_size, n)  # exclusive
        # char span: from offsets[start_token][0] to offsets[end_token-1][1]
        char_start = offsets[start_token][0]
        char_end = offsets[end_token - 1][1] if end_token - 1 < len(offsets) else len(text)
        chunk_text = text[char_start:char_end]
        chunks.append(chunk_text)
        start_token += stride

    return chunks
