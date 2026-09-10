import numpy as np
import os
from sentence_transformers import SentenceTransformer
import faiss

def file_loader():
    p = r"data"

    articles = []
    with os.scandir(p) as entries:
        for e in entries:
            if e.is_file() and e.name.endswith(".txt"):
                with open(e.path, "r", encoding="utf-8") as f:
                    text = f.read()

                articles.append({
                    "source": e.name,
                    "text": text
                })
    return articles

def recursive_chunking(text: str, max_chunk_size: int = 1000):
    # Base case: if text is small enough, return as single chunk
    if len(text) <= max_chunk_size:
        return [text.strip()] if text.strip() else []

    # Try separators in priority order
    separators = ["\n\n", "\n", ". ", " "]

    for separator in separators:
        if separator in text:
            parts = text.split(separator)
            chunks = []
            current_chunk = ""

            for part in parts:
                # Check if adding this part would exceed the limit
                test_chunk = current_chunk + separator + part if current_chunk else part

                if len(test_chunk) <= max_chunk_size:
                    current_chunk = test_chunk
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = part

            # Add the final chunk
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Recursively process any chunks that are still too large
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > max_chunk_size:
                    final_chunks.extend(recursive_chunking(chunk, max_chunk_size))
                else:
                    final_chunks.append(chunk)

            return [chunk for chunk in final_chunks if chunk]

    # Fallback: split by character limit if no separators work
    return [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

def embed(list_of_sentences: list):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(list_of_sentences)

    return embeddings

def build_index(embeddings: np.ndarray):
    # Compares against every stored vector, no approximation, exact results.
    # "L2" is straight-line distance.
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def main():
    articles = file_loader()
    all_chunks = []
    for article in articles:
        chunks = recursive_chunking(article['text'])
        for i, chunk in enumerate(chunks):
            all_chunks.append({"source": article['source'], "text": chunk, "index": i})
    embeddings =embed([chunk['text'] for chunk in all_chunks])
    vectors = build_index(embeddings)
    print(vectors)

main()
