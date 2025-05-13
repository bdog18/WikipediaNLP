import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import gradio as gr
import os

# Load SentenceTransformer model
# model = SentenceTransformer("all-MiniLM-L12-v2")

# Try to load FAISS index
index_path = "../data/embeddings/faiss_index.index"
index = None
if os.path.isfile(index_path):
    index = faiss.read_index(index_path)
else:
    print(f"⚠️ FAISS index not found at {index_path}. Backend search will be disabled.")

# Try to load article metadata
metadata_path = "../data/embeddings/article_metadata.json"
metadata = None
if os.path.isfile(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    print(f"⚠️ Metadata JSON not found at {metadata_path}. Backend search will be disabled.")

# Define search function
def search_wiki(query, k=5):
    if index is None or metadata is None:
        return "⚠️ Backend data not available. You're in frontend-only mode."
    # query_embedding = model.encode([query])
    # D, I = index.search(np.array(query_embedding).astype('float32'), k)
    # results = []
    # for idx in I[0]:
    #     meta = metadata[idx]
    #     title = meta.get("title", "Untitled")
    #     url = meta.get("url", "#")
    #     results.append(f"🔹 [{title}]({url})")
    # return "\n".join(results)
    return 

css = """
#custom-query-box, #custom-output-box {
    max-width: 600px;
    width: 70%;
    margin-left: auto;
    margin-right: auto;
}
"""
# Gradio interface
with gr.Blocks(css=css) as iface:
    with gr.Column():
        gr.Markdown("## 📚 Wikipedia Semantic Search")

        query_input = gr.Textbox(
            lines=2,
            placeholder="Enter your semantic query...",
            label="Search Query",
            elem_id="custom-query-box"
        )

        k_slider = gr.Slider(
            minimum=1,
            maximum=10,
            step=1,
            value=5,
            label="Top K Results"
        )

        output_box = gr.Markdown(elem_id="custom-output-box")

        query_input.change(fn=search_wiki, inputs=[query_input, k_slider], outputs=output_box)
        k_slider.change(fn=search_wiki, inputs=[query_input, k_slider], outputs=output_box)

if __name__ == "__main__":
    iface.launch()
