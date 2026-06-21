"""
results/scored_*.jsonl 들을 모아 시스템별 평균 비교표 + 막대그래프를 만든다.

  python eval/aggregate.py
출력: results/comparison.json, results/comparison.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

KEYS = ["bleurt_component_50", "judge_component_50", "final_score_100",
        "sentence_count_score", "signature_phrase_score", "semantic_similarity",
        "bleurt_norm", "content_quality", "persona_alignment", "rhetorical_alignment"]


def main():
    rows = []
    for fp in sorted(config.RESULTS_DIR.glob("scored_*.jsonl")):
        data = [json.loads(l) for l in open(fp, encoding="utf-8")]
        name = fp.stem.replace("scored_", "")
        agg = {"system": name, "n": len(data)}
        for k in KEYS:
            vals = [d[k] for d in data if d.get(k) is not None]
            agg[k] = round(sum(vals) / len(vals), 3) if vals else 0.0
        rows.append(agg)

    out = config.RESULTS_DIR / "comparison.json"
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"✅ {out}")
    print("\nsystem | n | BLEURT(50) | Judge(50) | Total(100)")
    for r in rows:
        print(f"{r['system']} | {r['n']} | {r['bleurt_component_50']} | "
              f"{r['judge_component_50']} | {r['final_score_100']}")

    # 막대그래프
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        labels = [r["system"] for r in rows]
        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2.2), 4.5))
        ax.bar(x - w / 2, [r["bleurt_component_50"] for r in rows], w, label="BLEURT/Rule (50)")
        ax.bar(x + w / 2, [r["judge_component_50"] for r in rows], w, label="LLM-Judge (50)")
        for i, r in enumerate(rows):
            ax.text(i, r["final_score_100"] + 1, f"{r['final_score_100']:.1f}",
                    ha="center", fontweight="bold")
        ax.set_ylabel("score"); ax.set_title("Chimchak Persona LLM — Evaluation")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 100); ax.legend()
        fig.tight_layout()
        png = config.RESULTS_DIR / "comparison.png"
        fig.savefig(png, dpi=130)
        print(f"✅ {png}")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 그래프 생략: {e}")


if __name__ == "__main__":
    main()
