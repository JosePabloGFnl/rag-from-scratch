import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

p = r"data"

articles = []
for e in os.scandir(p):
    if e.is_file() and e.name.endswith(".txt"):
        with open(e.path, "r", encoding="utf-8") as f:
            text = f.read()

            articles.append({
    "source": e.name,
    "text": text
})
