"""
RAG retriever — loads the prebuilt embedding cache (data/rag_cache.pkl) and does
in-memory cosine Top-k search over QA pairs (embed the question, return Q + A).

Cache structure (built by rag/build_index.py):
    {"ids": [...], "embeddings": [[...]], "documents": [user_text...],
     "metadatas": [{"assistant": ..., "source": ...}]}
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


class RagRetriever:
    def __init__(self, cache_path: Path = config.RAG_CACHE_PATH,
                 embed_model_name: str = config.EMBED_MODEL_NAME,
                 device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        if not cache_path.exists():
            raise FileNotFoundError(
                f"RAG cache not found: {cache_path}\n"
                f"Build it first: python rag/build_index.py"
            )
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

        self.documents: list[str] = cache["documents"]
        self.metadatas: list[dict] = cache["metadatas"]
        emb = np.asarray(cache["embeddings"], dtype=np.float32)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        self.embeddings = emb

        self.embed_model = SentenceTransformer(embed_model_name, device=device)
        print(f"✅ RAG cache loaded: {len(self.documents)} utterances, dim={emb.shape[1]}")

    def search(self, query: str, top_k: int = 2) -> list[dict]:
        q = np.asarray(self.embed_model.encode(query), dtype=np.float32)
        q /= (np.linalg.norm(q) + 1e-8)
        scores = self.embeddings @ q
        idx = np.argsort(-scores)[:top_k]
        return [
            {
                "user": self.documents[i],
                "assistant": self.metadatas[i].get("assistant", ""),
                "source": self.metadatas[i].get("source", "Unknown"),
                "score": float(scores[i]),
            }
            for i in idx
        ]
