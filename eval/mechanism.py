"""
메커니즘 분석 — 'FT가 왜 나쁜가'를 출력 통계로 규명.
각 시스템 answers_*.jsonl 에서:
  - avg_len  : 평균 글자수
  - distinct1/2 : 어휘 다양성(고유 uni/bi-gram 비율, 코퍼스 단위)
  - rep_rate : 답변 내 반복(중복 bigram 비율) 평균
  - signature : 침착맨 시그니처 표현 평균 포함수

  python eval/mechanism.py
출력: results/mechanism.json, results/mechanism.png
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

_persona = json.load(open(config.PERSONA_PROMPT_PATH, encoding="utf-8"))
_SIG = [s.strip() for s in (_persona["hard_data"]["starters"]
        + _persona["hard_data"]["fillers"]
        + _persona["hard_data"]["endings"]) if s.strip()]


def toks(t):
    return re.findall(r"[가-힣]+|[a-zA-Z]+|[0-9]+", t)


def analyze(rows):
    answers = [r["answer"] for r in rows]
    lens = [len(a) for a in answers]
    uni, bi = set(), set()
    tot_uni = tot_bi = 0
    rep = []
    sig = []
    for a in answers:
        ts = toks(a)
        tot_uni += len(ts)
        uni.update(ts)
        bg = list(zip(ts, ts[1:]))
        tot_bi += len(bg)
        bi.update(bg)
        rep.append(1 - len(set(bg)) / len(bg) if bg else 0.0)
        sig.append(sum(1 for s in _SIG if s in a))
    n = len(answers)
    return {
        "n": n,
        "avg_len": round(sum(lens) / n, 1),
        "distinct1": round(len(uni) / max(tot_uni, 1), 4),
        "distinct2": round(len(bi) / max(tot_bi, 1), 4),
        "rep_rate": round(sum(rep) / n, 4),
        "sig_per_ans": round(sum(sig) / n, 3),
    }


def main():
    out = {}
    for fp in sorted(config.RESULTS_DIR.glob("answers_*.jsonl")):
        name = fp.stem.replace("answers_", "")
        rows = [json.loads(l) for l in open(fp, encoding="utf-8")]
        out[name] = analyze(rows)

    json.dump(out, open(config.RESULTS_DIR / "mechanism.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"{'system':28s} {'len':>6} {'dist1':>7} {'dist2':>7} {'rep':>7} {'sig':>6}")
    for k, v in out.items():
        print(f"{k:28s} {v['avg_len']:>6} {v['distinct1']:>7} {v['distinct2']:>7} "
              f"{v['rep_rate']:>7} {v['sig_per_ans']:>6}")

    # 4-condition(Qwen3-4B) 다양성 막대그래프
    keys = [("base_Qwen3-4B", "Base"), ("ft_Qwen3-4B", "FT"),
            ("ragft_Qwen3-4B", "RAG+FT"), ("rag_Qwen3-4B", "RAG")]
    keys = [(k, l) for k, l in keys if k in out]
    if keys:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        labels = [l for _, l in keys]
        x = np.arange(len(keys))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].bar(x, [out[k]["sig_per_ans"] for k, _ in keys],
                  color=["#9aa5b1", "#e07a5f", "#f2cc8f", "#1f4e79"])
        ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
        ax[0].set_title("Persona signature phrases / answer  (cause)")
        ax[1].bar(x, [out[k]["distinct2"] for k, _ in keys],
                  color=["#9aa5b1", "#e07a5f", "#f2cc8f", "#1f4e79"])
        ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
        ax[1].set_title("Lexical diversity (distinct-2)  (rules out collapse)")
        fig.tight_layout()
        fig.savefig(config.RESULTS_DIR / "mechanism.png", dpi=140)
        print(f"\n✅ {config.RESULTS_DIR/'mechanism.png'}")


if __name__ == "__main__":
    main()
