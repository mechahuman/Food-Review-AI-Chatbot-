# restaurants_vectordb.py
from __future__ import annotations
import os
import math
import pandas as pd
from typing import Any, Dict

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

EXCEL_PATH = r"C:\HACKATHON\ollama_final\Restaurants.xlsx"
SHEET_NAME = 0
PERSIST_DIR = "./chroma_restaurants"
COLLECTION = "restaurants"
EMBED_MODEL = "nomic-embed-text"

def _load_df():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.fillna("")
    # Safe numeric coercion (no deprecation warnings)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    return df

def _row_to_text_and_meta(df: pd.DataFrame, row: pd.Series):
    pairs = []
    for col in df.columns:
        val = row[col]
        if isinstance(val, float) and math.isnan(val):
            continue
        pairs.append(f"{col}: {val}")
    page_content = " | ".join(pairs)

    meta = {"row_index": int(row.name)}
    for key in ["Restaurant","Cleanliness","Service","Pricing","Food/Drinks","Ambience","Overall","What_to_Try","Price_per_head","Location"]:
        if key in df.columns:
            meta[key] = row[key]
    return page_content, meta

def get_retriever(k: int = 4, search_type: str = "similarity"):
    # Fast path: only construct vector store on import; do not rebuild docs
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vector_store.as_retriever(
        search_type=search_type,           # "similarity" is faster than "mmr"
        search_kwargs={"k": k},
    )

def format_docs(docs, per_doc_char_limit: int = 600):
    blocks = []
    for i, d in enumerate(docs, 1):
        name = d.metadata.get("name") or d.metadata.get("restaurant") or d.metadata.get("title") or f"Restaurant #{i}"
        text = d.page_content[:per_doc_char_limit]
        blocks.append(f"---\n#{i} {name}\n{text}")
    return "\n".join(blocks)

if __name__ == "__main__":
    # Build/persist the DB only once (run this script once before chatting)
    print("Building restaurant vector DB (one-time)...")
    df = _load_df()
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    add_documents = not os.path.exists(PERSIST_DIR)

    vector_store = Chroma(
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    if add_documents:
        docs, ids = [], []
        for idx, row in df.iterrows():
            content, meta = _row_to_text_and_meta(df, row)
            docs.append(Document(page_content=content, metadata=meta))
            ids.append(str(idx))
        if docs:
            vector_store.add_documents(documents=docs, ids=ids)

    print("Done.")