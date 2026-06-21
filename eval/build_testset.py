"""
Build the held-out test set.

Samples N items from chim_qa_1.json, which is NOT in the RAG index
(data/rag_datasets/*.json) → avoids the retriever returning the gold answer.

  python eval/build_testset.py --n 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

SOURCE = config.TESTSET_SOURCE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(config.TESTSET_PATH))
    args = ap.parse_args()

    config.ensure_dirs()
    data = json.load(open(SOURCE, encoding="utf-8"))
    qa = data.get("qa_list", data) if isinstance(data, dict) else data
    qa = [x for x in qa if x.get("user") and x.get("assistant")]

    random.seed(args.seed)
    sample = random.sample(qa, min(args.n, len(qa)))

    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        for x in sample:
            f.write(json.dumps({
                "id": x.get("id"),
                "question": x["user"],
                "gold": x["assistant"],
                "topic": x.get("topic"),
            }, ensure_ascii=False) + "\n")
    print(f"✅ test set: {len(sample)} items → {out}")


if __name__ == "__main__":
    main()
