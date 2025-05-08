
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import gradio as gr
import os

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L12-v2")

# Load FAISS index
index_path = "../data/embeddings/faiss_index.index"
if not os.path.isfile(index_path):
    raise FileNotFoundError(f"FAISS index not found at {index_path}")
index = faiss.read_index(index_path)

# Load article metadata
metadata_path = "../data/embeddings/article_metadata.json"
if not os.path.isfile(metadata_path):
    raise FileNotFoundError(f"Metadata JSON not found at {metadata_path}")
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Define search function
def search_wiki(query, k=5):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding).astype('float32'), k)
    results = []
    for idx in I[0]:
        meta = metadata[idx]
        title = meta.get("title", "Untitled")
        url = meta.get("url", "#")
        results.append(f"🔹 [{title}]({url})")
    return "\n".join(results)

# Gradio interface
iface = gr.Interface(
    fn=search_wiki,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your semantic query...", label="Search Query"),
        gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Top K Results")
    ],
    outputs="markdown",
    title="📚 Wikipedia Semantic Search",
    description="Enter a natural language query to retrieve the most relevant Wikipedia articles using FAISS and Sentence Transformers."
)

if __name__ == "__main__":
    iface.launch()
