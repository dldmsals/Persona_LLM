"""
평가 오케스트레이터.

각 시스템(rag / ft / base)으로 테스트셋 답변을 생성 → 메트릭 + judge 채점 →
Total(100) = bleurt_component_50 + judge_component_50 집계 후 표/JSON 출력.

  # 1) 답변 생성 + 채점 (RAG)
  python eval/run_eval.py --systems rag --limit 100

  # 2) finetuned 로컬 모델 답변 채점 (어댑터 회수 후)
  python eval/run_eval.py --systems ft --ft_path finetune/out

  # 3) 이미 생성된 답변 파일 채점만
  python eval/run_eval.py --systems rag --score_only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from eval.metrics import Metrics  # noqa: E402
from eval.judge import Judge  # noqa: E402


def load_testset(limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in open(config.TESTSET_PATH, encoding="utf-8")]
    return rows[:limit] if limit else rows


def gen_rag(testset, model, top_k=2):
    from rag.chat import PersonaChat
    bot = PersonaChat(model_name=model, top_k=top_k)
    out = []
    for i, r in enumerate(testset, 1):
        ans = bot.generate(r["question"])
        out.append({**r, "answer": ans})
        print(f"  [rag {i}/{len(testset)}] {ans[:40]}...")
    return out


def gen_ft(testset, ft_path, base_model):
    """로컬 finetuned(QLoRA 머지/어댑터) 모델로 생성. GPU 필요."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from rag.chat import SYSTEM_TEMPLATE

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")
    if ft_path:
        model = PeftModel.from_pretrained(model, ft_path)
    model.eval()

    out = []
    sys_prompt = SYSTEM_TEMPLATE.format(few_shot="")
    for i, r in enumerate(testset, 1):
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": r["question"]}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        in_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=200, do_sample=True,
                                 temperature=0.7, top_p=0.9,
                                 pad_token_id=tok.eos_token_id)
        ans = tok.decode(gen[0][in_len:], skip_special_tokens=True).strip()
        out.append({**r, "answer": ans})
        print(f"  [ft {i}/{len(testset)}] {ans[:40]}...")
    return out


def score_rows(rows, use_bleurt=True, use_judge=True, judge_model=None):
    metrics = Metrics(use_bleurt=use_bleurt)
    judge = Judge(model=judge_model) if use_judge else None
    scored = []
    for i, r in enumerate(rows, 1):
        m = metrics.score(r["answer"], r["gold"])
        j = judge.score(r["question"], r["gold"], r["answer"]) if judge else {}
        final = m["bleurt_component_50"] + j.get("judge_component_50", 0.0)
        scored.append({**r, **m, **j, "final_score_100": round(final, 4)})
        print(f"  [score {i}/{len(rows)}] final={final:.2f}")
    return scored


def summarize(scored, system):
    n = len(scored)
    def avg(k):
        vals = [s[k] for s in scored if s.get(k) is not None]
        return sum(vals) / len(vals) if vals else 0.0
    return {
        "system": system, "n": n,
        "bleurt_component_50": round(avg("bleurt_component_50"), 3),
        "judge_component_50": round(avg("judge_component_50"), 3),
        "final_score_100": round(avg("final_score_100"), 3),
        "sentence_count_score": round(avg("sentence_count_score"), 3),
        "signature_phrase_score": round(avg("signature_phrase_score"), 3),
        "semantic_similarity": round(avg("semantic_similarity"), 3),
        "bleurt_norm": round(avg("bleurt_norm"), 3),
        "content_quality": round(avg("content_quality"), 3),
        "persona_alignment": round(avg("persona_alignment"), 3),
        "rhetorical_alignment": round(avg("rhetorical_alignment"), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="rag", help="쉼표구분: rag,ft,base")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--model", default=config.DEFAULT_RAG_MODEL)
    ap.add_argument("--judge_model", default=config.DEFAULT_JUDGE_MODEL)
    ap.add_argument("--ft_path", default=None)
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B")
    ap.add_argument("--no_bleurt", action="store_true")
    ap.add_argument("--no_judge", action="store_true")
    ap.add_argument("--score_only", action="store_true",
                    help="기존 answers_*.jsonl 재채점")
    args = ap.parse_args()

    config.ensure_dirs()
    testset = load_testset(args.limit)
    summaries = []

    for system in [s.strip() for s in args.systems.split(",") if s.strip()]:
        ans_path = config.RESULTS_DIR / f"answers_{system}_{args.model.split('/')[-1]}.jsonl"
        if args.score_only and ans_path.exists():
            rows = [json.loads(l) for l in open(ans_path, encoding="utf-8")]
        else:
            if system == "rag":
                rows = gen_rag(testset, args.model)
            elif system == "ft":
                rows = gen_ft(testset, args.ft_path, args.base_model)
            else:
                raise SystemExit(f"알 수 없는 system: {system}")
            with open(ans_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        scored = score_rows(rows, use_bleurt=not args.no_bleurt,
                             use_judge=not args.no_judge,
                             judge_model=args.judge_model)
        out_path = config.RESULTS_DIR / f"scored_{system}_{args.model.split('/')[-1]}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in scored:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        summaries.append(summarize(scored, f"{system}:{args.model}"))

    # 요약 표 출력 + 저장
    print("\n===== 요약 (평균) =====")
    keys = ["system", "n", "bleurt_component_50", "judge_component_50", "final_score_100"]
    print(" | ".join(keys))
    for s in summaries:
        print(" | ".join(str(s[k]) for k in keys))
    summary_path = config.RESULTS_DIR / "summary.json"
    json.dump(summaries, open(summary_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n✅ 요약 저장: {summary_path}")


if __name__ == "__main__":
    main()
