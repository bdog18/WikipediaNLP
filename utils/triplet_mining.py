import os
import sys
import json
import shelve
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from multiprocessing import Pool, cpu_count
import gc

# Add root directory (one level up from notebooks/)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from utils.sqlite_lookup import load_linkgraph_sqlite, get_links_for_title_sqlite

# CONFIGURATION
WIKI_DATA_DIR_JSONL = "../data/processed/wikidata_jsonl"
LINK_GRAPH_PATH = "../data/processed/wiki_link_graph.db"
PARAGRAPH_DB_PATH = "../data/processed/paragraphs_shelve.db"
TRIPLET_OUTPUT_DIR = "../data/processed/triplets/parallel_parts"
FAISS_INDEX_PATH = "../data/processed/faiss_index/paragraphs.index"
FAISS_META_PATH = FAISS_INDEX_PATH + ".meta.json"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TRIPLETS_PER_ARTICLE = 5
NEGATIVE_POOL_SIZE = 5

index = None
all_texts = None
text_to_meta = None
article_titles = None
model = None


def stream_article_lines(data_dir):
    files = [os.path.join(root, file)
             for root, _, filenames in os.walk(data_dir)
             for file in filenames if file.endswith(".json") or file.endswith(".jsonl")]
    random.shuffle(files)
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_faiss_index(index_path, meta_path):
    print("Loading FAISS index...")
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta["all_texts"], meta["text_to_meta"]


def load_articles_titles_only(data_dir):
    article_titles = {}
    files = [os.path.join(root, file)
             for root, _, filenames in os.walk(data_dir)
             for file in filenames if file.endswith(".jsonl")]
    for file_path in tqdm(files, desc="Loading Article Titles"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    article = json.loads(line)
                    title = article.get("title")
                    if title:
                        article_titles[title] = article.get("url", "")
                except json.JSONDecodeError:
                    continue
    return article_titles


def mine_triplets_from_file(file_path):
    local_model = SentenceTransformer(MODEL_NAME, device='cuda')
    link_conn = load_linkgraph_sqlite(LINK_GRAPH_PATH)
    para_db = shelve.open(PARAGRAPH_DB_PATH, flag='r')
    triplets = []
    global index, all_texts, text_to_meta, article_titles

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                article = json.loads(line)
                title = article.get("title")
                raw_text = article.get("content", "")
                paras = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
                linked_titles = get_links_for_title_sqlite(link_conn, title)

                if not linked_titles:
                    continue

                triplets_for_article = 0
                anchors, positives, meta = [], [], []

                for i, anchor in enumerate(paras[1:], 1):
                    if triplets_for_article >= MAX_TRIPLETS_PER_ARTICLE:
                        break
                    if not anchor.strip() or len(anchor) <= 20:
                        continue

                    positive = None
                    for linked_title in linked_titles:
                        linked_paras = para_db.get(linked_title, [])
                        positive = next((p for p in linked_paras if p.strip() and p.strip() != linked_title and linked_title.lower() not in p.lower() and len(p) > 20), None)
                        if positive:
                            break
                    if not positive and len(paras) > i + 1:
                        positive = paras[i + 1]
                    if not positive:
                        continue

                    anchors.append(anchor)
                    positives.append(positive)
                    meta.append((anchor, positive, title, linked_titles))
                    triplets_for_article += 1

                if not anchors:
                    continue

                anchor_embeddings = local_model.encode(anchors, convert_to_numpy=True, normalize_embeddings=True)
                D, I = index.search(anchor_embeddings, NEGATIVE_POOL_SIZE + 20)

                for idx, (anchor, positive, title, linked_titles) in enumerate(meta):
                    negative = None
                    for j in I[idx]:
                        neg_title = text_to_meta[j]
                        neg_para = all_texts[j]
                        if neg_title != title and neg_title not in linked_titles and neg_para != anchor and neg_para != positive and len(neg_para) > 20:
                            negative = neg_para
                            break
                    if not negative:
                        continue

                    triplets.append({
                        "anchor": anchor,
                        "positive": positive,
                        "negative": negative,
                        "source": title,
                        "url": article_titles.get(title, "")
                    })
            except json.JSONDecodeError:
                continue
    para_db.close()

    # Write per-process output
    rel_path = os.path.relpath(file_path, WIKI_DATA_DIR_JSONL)
    safe_name = rel_path.replace(os.sep, "__").replace(".jsonl", "_triplets.jsonl")
    output_path = os.path.join(TRIPLET_OUTPUT_DIR, safe_name)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for t in triplets:
            out_f.write(json.dumps(t, ensure_ascii=False) + "\n")
    
    gc.collect()
    return None

if __name__ == "__main__":
    os.makedirs(TRIPLET_OUTPUT_DIR, exist_ok=True)
    article_titles = load_articles_titles_only(WIKI_DATA_DIR_JSONL)
    index, all_texts, text_to_meta = load_faiss_index(FAISS_INDEX_PATH, FAISS_META_PATH)

    files = [os.path.join(root, file)
             for root, _, filenames in os.walk(WIKI_DATA_DIR_JSONL)
             for file in filenames if file.endswith(".jsonl")]

    with Pool(processes=4) as pool:
        results = list(tqdm(pool.imap_unordered(mine_triplets_from_file, files, chunksize=2), total=len(files), desc="Parallel Triplet Mining"))

    print(f"Triplet Writing Complete")
