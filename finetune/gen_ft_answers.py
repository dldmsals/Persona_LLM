"""
(GPU node) Generate test-set answers with a fine-tuned model → answers jsonl.
Standalone: no config / external API dependency.

  python gen_ft_answers.py --base Qwen/Qwen3-4B --adapter out_Qwen3-4B \
      --testset testset_100.jsonl --out answers_ft_Qwen3-4B.jsonl
"""
from __future__ import annotations

import argparse
import json

SYSTEM_PROMPT = (
    "당신은 인터넷 방송인 '침착맨(이말년)'입니다. 누군가 의견을 제시하면 특유의 킹받는 화법, "
    "기상천외한 비유, 그리고 뻔뻔하지만 묘하게 설득력 있는 '억지 논리'를 펼쳐서 상대방을 반박하세요. "
    "1~3문장으로 짧게 치고 빠지고, 구어체로 답하세요."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--testset", default="testset_100.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(args.testset, encoding="utf-8")]
    out = []
    for i, r in enumerate(rows, 1):
        # 행에 'system'(예: RAG few-shot 프롬프트)이 있으면 사용, 없으면 기본 프롬프트
        sys_prompt = r.get("system") or SYSTEM_PROMPT
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": r["question"]}]
        # Qwen3 thinking 모드 비활성화(공정 비교: 모든 조건 동일)
        try:
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True,
                                          enable_thinking=False)
        except TypeError:
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        in_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=True, temperature=0.7, top_p=0.9,
                                 pad_token_id=tok.eos_token_id)
        ans = tok.decode(gen[0][in_len:], skip_special_tokens=True).strip()
        out.append({**r, "answer": ans})
        if i % 10 == 0:
            print(f"  [{i}/{len(rows)}] {ans[:40]}...", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ 저장: {args.out} ({len(out)})")


if __name__ == "__main__":
    main()
