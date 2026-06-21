"""
Persona chatbot: retrieve Top-k past utterances, build a few-shot persona prompt,
and generate via an OpenAI-compatible provider (default OpenRouter).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from rag.retriever import RagRetriever  # noqa: E402

SYSTEM_TEMPLATE = """당신은 인터넷 방송인 '침착맨(이말년)'입니다.
당신의 목표는 아래 [예시]에서 침착맨이 **어떤 방식으로 억지 논리를 펼치는지(비유 방식, 회피 방식, 논리적 비약 등) 그 '사고방식과 논조'를 파악**하여,
현재 사용자의 질문에 그 억지 논리의 **패턴**을 적용해 대답하는 것입니다.

[예시]의 주어나 목적어를 그대로 베끼지 마세요! [예시]에 등장하는 '논리의 구조'나 '비유하는 방식'만을 훔쳐와서 새로운 상황(사용자의 질문)에 찰떡같이 맞아떨어지게 변형하세요.

[답변 절대 규칙]
1. 논리 차용: [예시]의 답변(A)이 가진 '핵심 억지 논리'를 파악하고, 그 패턴을 사용자의 질문에 맞춰 새롭게 변형하세요.
2. 길이 제한: 반드시 **1~3문장 이내**로 짧게 치고 빠지세요. (말이 길면 감점)
3. 태도: 상대방 말을 자르듯 뻔뻔하게 반박하고, 기상천외한 비유 하나만 툭 던지세요.
4. 구어체: "아니,", "그게 아니라,", "제가 조사를 다 했어요." 같은 말을 적극 활용하세요.

{few_shot}"""


class PersonaChat:
    def __init__(self, model_name: str = config.DEFAULT_RAG_MODEL,
                 retriever: RagRetriever | None = None, top_k: int = 2):
        self.model_name = model_name
        self.top_k = top_k
        self.retriever = retriever or RagRetriever()
        self.client = config.get_openai_client()

    def build_prompt(self, user_message: str) -> str:
        if self.top_k <= 0:  # no retrieval (k=0 point of the top-k sweep)
            return SYSTEM_TEMPLATE.format(few_shot="")
        hits = self.retriever.search(user_message, top_k=self.top_k)
        few_shot = ""
        for i, h in enumerate(hits, 1):
            few_shot += f"[예시 {i}]\nQ: {h['user']}\nA: {h['assistant']}\n\n"
        return SYSTEM_TEMPLATE.format(few_shot=few_shot)

    def generate(self, user_message: str, temperature: float = 0.7,
                 max_tokens: int = 200) -> str:
        system_prompt = self.build_prompt(user_message)
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def stream(self, user_message: str, temperature: float = 0.7,
               max_tokens: int = 200):
        system_prompt = self.build_prompt(user_message)
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
