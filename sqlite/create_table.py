#!/usr/bin/env python3
# sqlite_vectors_minimal.py
import sqlite3
import numpy as np
import os
import json
from typing import List, Tuple

DB = "./index_metadata.db"

from sklearn.metrics.pairwise import cosine_similarity

def cosine(a, b):
    return float(cosine_similarity([a], [b])[0][0])

def to_blob(vec: np.ndarray) -> bytes:
    # store as float32 bytes (compact)
    return vec.astype(np.float32).tobytes()

def from_blob(b: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).reshape((dim,))

def init_db():
    if os.path.exists(DB):
        os.remove(DB)
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
      CREATE TABLE docs (
        id TEXT PRIMARY KEY,
        content TEXT,
        vec BLOB,
        dim INTEGER,
        meta TEXT,
        model TEXT
      );
    """)
    conn.commit()
    conn.close()

def insert_doc(doc_id: str, content: str, vec: np.ndarray, meta: dict = None, model: str = "sample"):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO docs (id, content, vec, dim, meta, model)
      VALUES (?, ?, ?, ?, ?, ?)
    """, (doc_id, content, to_blob(vec), int(vec.shape[0]), json.dumps(meta or {}), model))
    conn.commit()
    conn.close()

def fetch_all() -> List[Tuple[str, str, bytes, int]]:
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT id, content, vec, dim FROM docs")
    rows = cur.fetchall()
    conn.close()
    return rows

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # handle zero vectors defensively
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))

def query_knn(query_vec: np.ndarray, k: int = 1):
    rows = fetch_all()
    dim = query_vec.shape[0]
    results = []
    for doc_id, content, blob, d in rows:
        if d != dim:
            # skip vectors of mismatched dimension
            continue
        vec = from_blob(blob, dim)
        score = cosine_similarity(query_vec, vec)
        results.append((score, doc_id, content))
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:k]

if __name__ == "__main__":
    # initialize db and insert two example vectors (length 3)
    init_db()

    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # doc1
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # doc2

    insert_doc("doc1", "Document about apples", v1, {"source": "notes"})
    insert_doc("doc2", "Document about bananas", v2, {"source": "notes"})

    # query vector (length 3)
    q = np.array([0.9, 0.1, 0.0], dtype=np.float32)

    results = query_knn(q, k=2)
    print("Top results (score, id, snippet):")
    for score, doc_id, content in results:
        print(f"{score:.6f}\t{doc_id}\t{content}")
