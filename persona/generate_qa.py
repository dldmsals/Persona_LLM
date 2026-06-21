"""
Persona-conditioned QA generation (sophistry-style answers).

Injects the persona profile (data/chim_persona_prompt.json) into a Gemini prompt
and generates {user, assistant} pairs over a pool of everyday themes.

Needs : GEMINI_API_KEY in .env
  python persona/generate_qa.py --n 1000
Output: data/chim_absurd_qa_1000.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

THEME_POOL = {
    "일상/가사": ["설거지 미루기", "수건 냄새", "형광등 갈기 귀찮음", "분리수거", "양말 뒤집어 벗기"],
    "회사/학교": ["출근길 지옥철", "투머치토커 상사", "의미 없는 회의", "점심 메뉴 통일", "연차 사유"],
    "인간관계": ["축의금", "읽씹과 안읽씹", "명절 잔소리", "소개팅 정적", "MBTI 과몰입"],
    "소비/돈": ["새벽 충동구매", "배달비", "로또 1등 망상", "중고거래 네고", "구독료 방치"],
    "디지털/IT": ["유튜브 알고리즘", "배터리 5%", "비밀번호 까먹음", "키보드 워리어", "알림 강박"],
    "식생활/미식": ["탕수육 부먹찍먹", "민트초코 논쟁", "라면 물 조절", "국밥 가성비", "치킨무 누락"],
    "과학/우주": ["외계인", "블랙홀", "시간여행 패러독스", "평행우주", "AI 지배"],
    "철학/윤리": ["테세우스의 배", "트롤리 딜레마", "통 속의 뇌", "무인도 생존", "자유의지"],
    "극한 밸런스": ["평생 라면 vs 평생 고기", "와이파이 1년 vs 에어컨 1년", "평생 겨울 vs 여름"],
    "취미/여가": ["취미 장비병", "주말 순삭", "캠핑 텐트", "넷플릭스 고르다 잠듦", "아무것도 안 할 권리"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=50)
    args = ap.parse_args()

    import google.generativeai as genai
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is missing in .env")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")

    persona = json.load(open(config.PERSONA_PROMPT_PATH, encoding="utf-8"))
    save_path = config.ABSURD_QA_PATH
    out = []
    if save_path.exists():
        out = json.load(open(save_path, encoding="utf-8"))

    themes = list(THEME_POOL.keys())
    for i in range(len(out), args.n):
        theme = random.choice(themes)
        keyword = random.choice(THEME_POOL[theme])
        prompt = f"""당신은 아래 페르소나를 가진 화자입니다.
[수사 전략] {json.dumps(persona['rhetorical_strategy'], ensure_ascii=False)}
[핵심 로직] {json.dumps(persona['core_logic'], ensure_ascii=False)}
[시동어구/추임새] {persona['hard_data']['starters']} / {persona['hard_data']['fillers']}
[종결어미] {persona['hard_data']['endings']}

[미션] 테마={theme}, 키워드={keyword}
1. user: 위 키워드로 현대인이 겪을 법한 진지한 질문을 작성.
2. assistant: 질문과 무관한 '하찮은 일상 사물'을 궤변의 도구로 끌어와 답변.
3. 분량: 공백 포함 100~150자(최대 3문장). 뻔뻔하게 치고 빠지기.
[JSON] {{"id":"qa_{str(i+1).zfill(4)}","user":"...","assistant":"...","source_video":"none"}}"""
        try:
            resp = model.generate_content(prompt, generation_config=genai.GenerationConfig(
                response_mime_type="application/json", temperature=0.95))
            item = json.loads(resp.text)
            out.append(item)
            time.sleep(4.5)
            if (i + 1) % args.batch == 0:
                json.dump(out, open(save_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                print(f"💾 saved {i+1}")
        except Exception as e:  # noqa: BLE001
            print(f"retry ({e})"); time.sleep(10)

    json.dump(out, open(save_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"✅ done: {len(out)} → {save_path}")


if __name__ == "__main__":
    main()
