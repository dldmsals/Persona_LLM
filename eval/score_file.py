"""
이미 생성된 answers jsonl(예: 클러스터에서 만든 FT 답변)을 eb에서 채점.
BLEURT(CPU) + Rule + LLM-Judge(OpenRouter) → scored_{name}.jsonl + 요약 출력.

  python eval/score_file.py --answers results/answers_ft_Qwen3-4B.jsonl --name ft_Qwen3-4B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from eval.run_eval import score_rows, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--name", required=True, help="결과 라벨 (예: ft_Qwen3-4B)")
    ap.add_argument("--judge_model", default=config.DEFAULT_JUDGE_MODEL)
    ap.add_argument("--no_judge", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    rows = [json.loads(l) for l in open(args.answers, encoding="utf-8")]
    scored = score_rows(rows, use_bleurt=True, use_judge=not args.no_judge,
                        judge_model=args.judge_model)
    out = config.RESULTS_DIR / f"scored_{args.name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    summ = summarize(scored, args.name)
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    print(f"✅ {out}")


if __name__ == "__main__":
    main()
