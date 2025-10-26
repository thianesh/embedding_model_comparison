#!/usr/bin/env python3
# sqlite_vectors_minimal.py
import sqlite3
import numpy as np
import os
import json
from typing import List, Tuple

DB = "./sqlite/index_metadata.db"

# from sklearn.metrics.pairwise import cosine_similarity

# def cosine(a, b):
#     return float(cosine_similarity([a], [b])[0][0])

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # handle zero vectors defensively
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))

def from_blob(b: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).reshape((dim,))

def query_knn(query_vec: np.ndarray, k: int = 1, model: str = "sample"):
    rows = fetch_all(model)
    dim = query_vec.shape[0]
    results = []
    for doc_id, content, blob, d, meta, model in rows:
        if d != dim:
            # skip vectors of mismatched dimension
            continue
        vec = from_blob(blob, dim)
        score = cosine_similarity(query_vec, vec)
        results.append((score, doc_id, content, meta, model))
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:k]

def fetch_all(model: str) -> List[Tuple[str, str, bytes, int]]:
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT id, content, vec, dim, meta, model FROM docs WHERE model = ?"
                , (model,))
    rows = cur.fetchall()
    conn.close()
    return rows

def delete_all_rows(model):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("DELETE * FROM docs WHERE model = ?"
                , (model,))
    conn.commit()  # Commit the deletion
    conn.close()
    return True

def to_blob(vec: np.ndarray) -> bytes:
    # store as float32 bytes (compact)
    return vec.astype(np.float32).tobytes()

def insert_doc(doc_id: str, content: str, vec: np.ndarray, meta: dict = None, model: str = "sample"):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO docs (id, content, vec, dim, meta, model)
      VALUES (?, ?, ?, ?, ?, ?)
    """, (doc_id, content, to_blob(vec), int(vec.shape[0]), json.dumps(meta or {}), model))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # query vector (length 3)
    q = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    results = query_knn(q, k=2, model="sample")
    print("Top results (score, id, snippet):")
    for score, doc_id, content, meta, model in results:
        print(f"{score:.6f}\t{doc_id}\t{content}")