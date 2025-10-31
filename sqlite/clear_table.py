import sqlite3
from typing import List, Tuple

DB = "./sqlite/index_metadata.db"


def clear_all(table: str):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {table}")
    conn.commit()  # Important: persist the changes
    conn.close()

def fetch_all(model: str) -> List[Tuple[str, str, bytes, int]]:
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT id, content, vec, dim, meta, model FROM docs WHERE model = ?"
                , (model,))
    rows = cur.fetchall()
    conn.close()
    return rows

def fetch_everything() -> List[Tuple[str, str, bytes, int]]:
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT * FROM docs")
    rows = cur.fetchall()
    conn.close()
    return rows

# print(fetch_everything())
print(clear_all("docs"))
print(fetch_everything())