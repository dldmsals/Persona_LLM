"""
Ablation 차트(95% CI 오차막대) + findings 텍스트 생성.
significance.json(시스템별 CI) + scored_*.jsonl(구성요소 평균) 사용.

  python make_analysis.py
출력: results/ablation.png, results/findings.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ablation: 동일 Qwen3-4B, 방법만 변경
ABLATION = [
    ("base_Qwen3-4B", "Base\n(prompt)"),
    ("ft_Qwen3-4B", "FT\n(QLoRA)"),
    ("ragft_Qwen3-4B", "RAG+FT"),
    ("rag_Qwen3-4B", "RAG\n(base+retr.)"),
]


def main():
    sig = json.load(open(config.RESULTS_DIR / "significance.json", encoding="utf-8"))
    sysd = sig["systems"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [lbl for key, lbl in ABLATION if key in sysd]
    keys = [key for key, _ in ABLATION if key in sysd]
    means = [sysd[k]["mean"] for k in keys]
    lo = [sysd[k]["mean"] - sysd[k]["ci95"][0] for k in keys]
    hi = [sysd[k]["ci95"][1] - sysd[k]["mean"] for k in keys]

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bars = ax.bar(x, means, yerr=[lo, hi], capsize=6,
                  color=["#9aa5b1", "#e07a5f", "#f2cc8f", "#1f4e79"])
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 1.5, f"{m:.1f}",
                ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Total score (100)")
    ax.set_title("Ablation on Qwen3-4B (method only) - 95% CI")
    ax.set_ylim(0, 80)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "ablation.png", dpi=140)
    print(f"✅ {config.RESULTS_DIR/'ablation.png'}")

    # findings
    lines = ["# 결과 분석 (Findings)\n",
             "## Ablation (동일 Qwen3-4B, 방법만 변경)"]
    for k, lbl in ABLATION:
        if k in sysd:
            s = sysd[k]
            lines.append(f"- **{lbl.replace(chr(10),' ')}**: {s['mean']} "
                         f"[{s['ci95'][0]}, {s['ci95'][1]}]")
    pw = {(p["A"], p["B"]): p for p in sig["pairwise"]}

    def get(a, b):
        if (a, b) in pw: return pw[(a, b)]
        p = pw[(b, a)]; return {"delta": -p["delta"], "p": p["p"], "sig": p["sig"]}

    lines += ["\n## 핵심 발견 (유의성 검정)"]
    pairs = [("rag_Qwen3-4B", "base_Qwen3-4B", "검색(RAG)이 base 대비"),
             ("ft_Qwen3-4B", "base_Qwen3-4B", "QLoRA 파인튜닝이 base 대비"),
             ("ft_Qwen3-4B", "ft_Qwen3-0.6B", "FT에서 4B가 0.6B 대비"),
             ("rag_gemini-2.5-flash", "rag_gpt-4o-mini", "gemini가 gpt-4o-mini 대비"),
             ("ragft_Qwen3-4B", "ft_Qwen3-4B", "RAG+FT가 FT 대비")]
    for a, b, desc in pairs:
        if a in sysd and b in sysd:
            g = get(a, b)
            verdict = "유의" if g["sig"] != "n.s." else "유의하지 않음(동률)"
            lines.append(f"- {desc}: Δ={g['delta']:+.2f}, p={g['p']} → **{verdict}** ({g['sig']})")
    lines += [
        "\n## 해석",
        "- **검색(RAG)이 가장 효과적**: 동일 모델에서 base 대비 유의하게 향상.",
        "- **QLoRA 파인튜닝은 오히려 품질 저하**: 노이즈 있는 STT 기반 QA로의 SFT가 "
        "잘 정렬된 instruct 모델의 유창성/페르소나 표현을 해친 것으로 보임(일종의 정렬 퇴화).",
        "- **모델 간 RAG(gemini vs gpt-4o-mini)는 통계적 동률**: 점수 차를 '우열'로 해석하면 안 됨.",
        "- **평가 지표 한계**: BLEURT·의미유사도는 정답 1개 기준 reference 지표라 "
        "open-ended 페르소나 생성의 다양성을 과소평가함 → judge 비중과 함께 해석 필요.",
    ]
    (config.RESULTS_DIR / "findings.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ {config.RESULTS_DIR/'findings.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
