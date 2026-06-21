"""
LLM-as-a-Judge (정성 평가) — OpenRouter 경유.

생성된 답변을 세 축으로 0~1 채점:
  - content_quality      : 질문에 대한 내용 정합성/설득력
  - persona_alignment    : 침착맨 캐릭터(말투/뻔뻔함/억지논리) 일치도
  - rhetorical_alignment : 기상천외한 비유·프레임 전환 등 수사 전략 반영도
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

JUDGE_SYSTEM = """당신은 한국어 페르소나 챗봇 출력을 평가하는 엄격한 심사위원입니다.
평가 대상은 인터넷 방송인 '침착맨(이말년)'의 말투/사고를 모사한 답변입니다.
침착맨 특징: 짧고 뻔뻔한 반박, 기상천외한 비유, 억지 논리, 구어체("아니,", "~잖아요").

아래 세 항목을 각각 0.0~1.0 사이 실수로 평가하고, 반드시 JSON만 출력하세요.
{
  "content_quality": <0~1>,
  "persona_alignment": <0~1>,
  "rhetorical_alignment": <0~1>,
  "reason": "<한 줄 근거>"
}"""

JUDGE_USER = """[질문]
{question}

[정답 예시(침착맨 실제 발화)]
{gold}

[평가 대상 답변]
{answer}

위 답변을 평가해 JSON으로만 답하세요."""


class Judge:
    def __init__(self, model: str = config.DEFAULT_JUDGE_MODEL):
        self.model = model
        self.client = config.get_openai_client()  # OpenRouter

    def score(self, question: str, gold: str, answer: str) -> dict:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": JUDGE_USER.format(
                        question=question, gold=gold, answer=answer)},
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            return {"content_quality": 0.0, "persona_alignment": 0.0,
                    "rhetorical_alignment": 0.0, "judge_reason": f"error: {e}",
                    "judge_avg": 0.0, "judge_component_50": 0.0}

        cq = float(data.get("content_quality", 0))
        pa = float(data.get("persona_alignment", 0))
        ra = float(data.get("rhetorical_alignment", 0))
        avg = (cq + pa + ra) / 3.0
        return {
            "content_quality": round(cq, 3),
            "persona_alignment": round(pa, 3),
            "rhetorical_alignment": round(ra, 3),
            "judge_reason": data.get("reason", ""),
            "judge_avg": round(avg, 4),
            "judge_component_50": round(50 * avg, 4),
        }
