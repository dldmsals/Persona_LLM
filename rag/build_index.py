"""
Build the RAG index: embed every QA pair's question (`user`) with a Korean
sentence embedding model and store an in-memory pickle cache.

  python rag/build_index.py
Output: data/rag_cache.pkl
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def build():
    from sentence_transformers import SentenceTransformer
    print("🧠 Building RAG index ...")
    embed_model = SentenceTransformer(config.EMBED_MODEL_NAME, device="cpu")

    all_qa = []

    # claims file: expand each claim into a QA-like entry
    if config.CLAIMS_PATH.exists():
        claims = json.load(open(config.CLAIMS_PATH, encoding="utf-8"))
        for entry in claims:
            source = entry.get("source", "Unknown")
            topic = entry.get("topic", "")
            for claim in entry.get("claims", []):
                if isinstance(claim, str) and claim.strip():
                    user_text = f"[{topic}] {claim}" if topic else claim
                    all_qa.append({"id": f"claim_{source}", "user": user_text,
                                   "assistant": claim, "source_video": source})

    for fp in sorted(glob.glob(str(config.RAG_DATASETS_DIR / "*.json"))):
        data = json.load(open(fp, encoding="utf-8"))
        qa_list = data.get("qa_list", data) if isinstance(data, dict) else data
        valid = [x for x in qa_list if isinstance(x, dict) and "user" in x and "assistant" in x]
        skipped = len(qa_list) - len(valid)
        print(f"  {os.path.basename(fp)}: +{len(valid)} (skipped {skipped})")
        all_qa.extend(valid)

    print(f"⏳ Embedding {len(all_qa)} questions ...")
    texts = [x["user"] for x in all_qa]
    embeddings = embed_model.encode(texts, show_progress_bar=True).tolist()
    metadatas = [{"assistant": x["assistant"], "source": x.get("source_video") or "Unknown"}
                 for x in all_qa]
    ids = [f'{x["id"]}_{i}' for i, x in enumerate(all_qa)]

    with open(config.RAG_CACHE_PATH, "wb") as f:
        pickle.dump({"ids": ids, "embeddings": embeddings,
                     "documents": texts, "metadatas": metadatas}, f)
    print(f"✅ Saved {len(all_qa)} entries → {config.RAG_CACHE_PATH}")


if __name__ == "__main__":
    build()
