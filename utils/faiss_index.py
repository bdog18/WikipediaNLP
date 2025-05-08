import faiss
import numpy as np
import os
from tqdm import tqdm


def create_faiss_index(faiss_idx_path, article_embedding_path):
    if not os.path.isfile(faiss_idx_path):
        # Load the embeddings and Initialize the FAISS index
        embeddings = np.load(article_embedding_path)
        index = faiss.IndexFlatL2(embeddings.shape[1])

        # Add embeddings in batches
        batch_size = 1000
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Adding to FAISS index"):
            batch = embeddings[i:i+batch_size]
            index.add(batch)

        faiss.write_index(index, faiss_idx_path)
        print("FAISS index saved.")
    else:
        print("FAISS index already exists.")

def query_faiss(faiss_idx_path, query_embedding, k):
    # Load FAISS index
    index = faiss.read_index(faiss_idx_path)

    # Search FAISS (returns distances and indices)
    _, indices = index.search(query_embedding, k)
    return indices