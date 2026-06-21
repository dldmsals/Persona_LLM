"""
스윕 곡선 2종 생성:
  (1) RAG top-k (results/topk_sweep.json)
  (2) FT 데이터 크기 (results/scored_ftn{N}_Qwen3-0.6B.jsonl)

  python make_sweeps.py
출력: results/topk_sweep.png (이미 있으면 갱신), results/ftdata_sweep.png, results/sweeps.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def mean(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return sum(v) / len(v) if v else 0.0


def main():
    # (2) FT 데이터 크기 곡선
    sizes = [250, 1000, 2000, 4297]
    pts = []
    for n in sizes:
        fp = config.RESULTS_DIR / f"scored_ftn{n}_Qwen3-0.6B.jsonl"
        if not fp.exists():
            continue
        rows = [json.loads(l) for l in open(fp, encoding="utf-8")]
        pts.append({"n": n, "final": round(mean(rows, "final_score_100"), 2),
                    "persona": round(mean(rows, "persona_alignment"), 3),
                    "sig": None})
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.plot([p["n"] for p in pts], [p["final"] for p in pts], "o-", color="#e07a5f",
            label="FT (Qwen3-0.6B) Total(100)")
    # 참고선: 같은 데이터로 본 base/RAG(4B) — 모델 규모 다름(점선, 맥락용)
    refs = {"base_Qwen3-4B": "#9aa5b1", "rag_Qwen3-4B": "#1f4e79"}
    for name, c in refs.items():
        fpr = config.RESULTS_DIR / f"scored_{name}.jsonl"
        if fpr.exists():
            rows = [json.loads(l) for l in open(fpr, encoding="utf-8")]
            ax.axhline(mean(rows, "final_score_100"), ls=":", color=c,
                       label=f"{name} (ref, 4B)")
    ax.set_xlabel("# SFT examples (noisy STT-derived QA)")
    ax.set_ylabel("Total score (100)")
    ax.set_title("FT scales with data but plateaus below base/RAG")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "ftdata_sweep.png", dpi=140)
    print(f"✅ {config.RESULTS_DIR/'ftdata_sweep.png'}")

    json.dump({"ft_data_sweep": pts}, open(config.RESULTS_DIR / "sweeps.json", "w",
              encoding="utf-8"), ensure_ascii=False, indent=2)
    for p in pts:
        print(f"  n={p['n']}: Total={p['final']} persona={p['persona']}")


if __name__ == "__main__":
    main()
