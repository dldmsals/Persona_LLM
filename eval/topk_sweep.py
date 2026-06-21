"""
RAG top-k 스윕 — 검색량(k)에 따른 성능 곡선. (동일 모델 고정)
k=0 은 검색 없음(base 프롬프트).

  python eval/topk_sweep.py --model google/gemini-2.5-flash --ks 0,1,2,4,8
출력: results/scored_ragk{K}_{model}.jsonl, results/topk_sweep.json + topk_sweep.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from eval.run_eval import gen_rag, score_rows, summarize, load_testset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.DEFAULT_RAG_MODEL)
    ap.add_argument("--ks", default="0,1,2,4,8")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    config.ensure_dirs()
    testset = load_testset(args.limit)
    ks = [int(x) for x in args.ks.split(",")]
    mtag = args.model.split("/")[-1]
    curve = []
    for k in ks:
        print(f"\n===== top_k={k} =====")
        rows = gen_rag(testset, args.model, top_k=k)
        scored = score_rows(rows, use_bleurt=True, use_judge=True,
                            judge_model=config.DEFAULT_JUDGE_MODEL)
        name = f"ragk{k}_{mtag}"
        with open(config.RESULTS_DIR / f"scored_{name}.jsonl", "w", encoding="utf-8") as f:
            for s in scored:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        summ = summarize(scored, name)
        summ["top_k"] = k
        curve.append(summ)
        print(f"k={k}: Total={summ['final_score_100']}")

    json.dump(curve, open(config.RESULTS_DIR / "topk_sweep.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [c["top_k"] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.plot(xs, [c["final_score_100"] for c in curve], "o-", label="Total(100)")
    ax.plot(xs, [c["bleurt_component_50"] for c in curve], "s--", label="BLEURT/Rule(50)")
    ax.plot(xs, [c["judge_component_50"] for c in curve], "^--", label="Judge(50)")
    ax.set_xlabel("RAG top-k (0 = no retrieval)"); ax.set_ylabel("score")
    ax.set_title(f"RAG retrieval-amount sweep ({mtag})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "topk_sweep.png", dpi=140)
    print(f"\n✅ {config.RESULTS_DIR/'topk_sweep.png'}")


if __name__ == "__main__":
    main()
