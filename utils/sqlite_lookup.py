import sqlite3
import json
import os
from tqdm import tqdm

def build_linkgraph_sqlite(jsonl_dir, db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS links (from_title TEXT PRIMARY KEY, linked_titles TEXT)")

    for root, _, files in os.walk(jsonl_dir):
        for file in tqdm(files, desc="Building SQLite"):
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            cur.execute(
                                "INSERT INTO links (from_title, linked_titles) VALUES (?, ?)",
                                (entry["from_title"], json.dumps(entry["linked_titles"]))
                            )
                        except Exception:
                            continue

    conn.commit()
    conn.close()

def load_linkgraph_sqlite(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def get_links_for_title_sqlite(conn, title):
    cur = conn.cursor()
    cur.execute("SELECT linked_titles FROM links WHERE from_title = ?", (title,))
    row = cur.fetchone()
    if row:
        cleaned = row["linked_titles"].replace("\\'", "'").replace("\\\'", "'")
        return set(json.loads(cleaned))
    return set()

if __name__ == "__main__":
    WIKI_LINK_GRAPH_JSONL_PATH = r"../data/processed/wiki_link_graph_jsonl"
    WIKI_LINK_GRAPH_DB_PATH = r"../data/processed/wiki_link_graph.db"

    # build_linkgraph_sqlite(WIKI_LINK_GRAPH_JSONL_PATH, WIKI_LINK_GRAPH_DB_PATH)
    conn = load_linkgraph_sqlite(WIKI_LINK_GRAPH_DB_PATH)
    row = get_links_for_title_sqlite(conn, "Heroic couplet")
    print(row)