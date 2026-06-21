"""
Build QLoRA SFT training data (messages-format jsonl).

Source: persona QA datasets. The test source (chim_qa_1.json) is excluded to
prevent leakage. Each record: {"messages":[{system},{user},{assistant}]}

  python finetune/prepare_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

SYSTEM_PROMPT = (
    "당신은 인터넷 방송인 '침착맨(이말년)'입니다. 누군가 의견을 제시하면 특유의 킹받는 화법, "
    "기상천외한 비유, 그리고 뻔뻔하지만 묘하게 설득력 있는 '억지 논리'를 펼쳐서 상대방을 반박하세요. "
    "1~3문장으로 짧게 치고 빠지고, 구어체로 답하세요."
)

# training sources (test source chim_qa_1.json intentionally excluded)
SOURCES = [
    config.RAG_DATASETS_DIR / "chim_qa_2.json",
    config.RAG_DATASETS_DIR / "chim_qa_3.json",
    config.RAG_DATASETS_DIR / "chim_qa_4.json",
    config.ABSURD_QA_PATH,
]


def iter_pairs(path: Path):
    if not path.exists():
        print(f"⚠️  skip (missing): {path.name}")
        return
    data = json.load(open(path, encoding="utf-8"))
    qa = data.get("qa_list", data) if isinstance(data, dict) else data
    for x in qa:
        if isinstance(x, dict) and x.get("user") and x.get("assistant"):
            yield x["user"].strip(), x["assistant"].strip()


def main():
    config.ensure_dirs()
    seen = set()
    n = 0
    out_path = config.SFT_DATA_PATH
    with open(out_path, "w", encoding="utf-8") as f:
        for src in SOURCES:
            cnt = 0
            for user, assistant in iter_pairs(src):
                key = (user, assistant)
                if key in seen:
                    continue
                seen.add(key)
                rec = {"messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cnt += 1
                n += 1
            print(f"  {src.name}: {cnt}")
    print(f"✅ SFT data: {n} examples → {out_path}")


if __name__ == "__main__":
    main()
