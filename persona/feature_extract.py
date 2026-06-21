"""
Persona feature extraction → data/chim_persona_prompt.json

Two kinds of features from the speaker's transcribed utterances:
  (A) Hard data  : Kiwi morphological statistics of verbal tics
                   (top trigram starters / IC fillers / sentence endings)
  (B) Soft data  : Gemini distills the rhetorical strategy into a fixed JSON schema
                   (rhetorical_strategy + core_logic)

Input : data/raw_transcripts/*.txt   (lines like "[..] SPEAKER_00: <utterance>")
Needs : GEMINI_API_KEY in .env

  python persona/feature_extract.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

TRANSCRIPTS_DIR = config.DATA_DIR / "raw_transcripts"
TARGET_SPEAKER = "SPEAKER_00"

# progress/host-specific phrases to exclude from tic statistics
NOISE = {'오른쪽', '왼쪽', '가도록 하겠습니다', '하겠습니다', '이번에', '강', '진출',
         '우승', '이상형', '월드컵', '라운드', '결승', '올라갑니다', '올라갈게요',
         '엉덩이', '테이블', '개판치고', '승자'}
STOP = {'아', '어', '그', '뭐', '자', '음', '예', '네', '이제', '좀', '저', '이'}
COMMON_ENDINGS = {'어요', '는데', '습니다', '은데', '네요', '니다', '이다', '한다', '합니다'}


def clean(text: str) -> str:
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^가-힣\s\?!\.,~]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def main():
    from kiwipiepy import Kiwi
    kiwi = Kiwi(typos='basic')
    pat = re.compile(r'\[.*?\]\s*(SPEAKER_\d+)\s*:\s*(.*)')

    sentences, full_text = [], ""
    if not TRANSCRIPTS_DIR.exists():
        raise SystemExit(f"transcripts not found: {TRANSCRIPTS_DIR}")
    for fp in TRANSCRIPTS_DIR.glob("*.txt"):
        for line in fp.read_text(encoding="utf-8").splitlines():
            m = pat.search(line)
            if m and m.group(1) == TARGET_SPEAKER:
                c = clean(m.group(2))
                if len(c) > 5:
                    sentences.append(c)
                    full_text += f"{c}\n"

    starters, endings, fillers = [], [], []
    for s in sentences:
        if any(w in s for w in NOISE):
            continue
        words = s.split()
        if len(words) >= 3 and not all(w in STOP for w in words[:3]):
            starters.append(" ".join(words[:3]))
        cs = re.sub(r'[^\w\s]', '', s).strip()
        if len(cs) >= 2 and cs[-2:] not in COMMON_ENDINGS:
            endings.append(cs[-3:] if len(cs) >= 3 else cs[-2:])
        for tok in kiwi.tokenize(s):
            if tok.tag == 'IC' and tok.form not in STOP:
                fillers.append(tok.form)

    hard = {
        "starters": [w for w, _ in Counter(starters).most_common(5)],
        "fillers": [w for w, _ in Counter(fillers).most_common(5)],
        "endings": [w for w, _ in Counter(endings).most_common(8)],
    }

    import google.generativeai as genai
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is missing in .env")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")
    schema = {"type": "OBJECT", "properties": {
        "rhetorical_strategy": {"type": "OBJECT", "properties": {
            "frame_redefinition": {"type": "STRING"},
            "improvised_immersion": {"type": "STRING"},
            "tonality_shift": {"type": "STRING"}}},
        "core_logic": {"type": "ARRAY", "items": {"type": "STRING"}}}}
    resp = model.generate_content(
        "제공된 발화 데이터를 분석하여 화자만의 고유 화법(평범한 소재를 기괴한 "
        f"메타포로 연결하는 비약 패턴 위주)을 분석하세요.\n[데이터]: {full_text[:15000]}",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json", response_schema=schema))
    analysis = json.loads(resp.text)

    out = {"persona_name": "침착맨", "hard_data": hard,
           "rhetorical_strategy": analysis["rhetorical_strategy"],
           "core_logic": analysis["core_logic"]}
    config.PERSONA_PROMPT_PATH.write_text(
        json.dumps(out, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"✅ saved: {config.PERSONA_PROMPT_PATH}")


if __name__ == "__main__":
    main()
