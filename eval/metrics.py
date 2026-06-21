"""
Quantitative metrics (BLEURT + rule-based).

Components (0~1):
  - sentence_count_score : 생성/정답 문장 수 유사도
  - signature_phrase_score : 침착맨 시그니처 표현(starters/fillers/endings) 반영도
  - semantic_similarity : ko-sroberta 코사인 (생성 vs 정답)
  - bleurt_norm : BLEURT 점수를 [0,1]로 정규화 (Elron/bleurt-tiny-128, CPU)

집계:
  heuristic_subavg = mean(sentence_count, signature_phrase, semantic_similarity)
  bleurt_rule_score = 0.4*bleurt_norm + 0.6*heuristic_subavg
  bleurt_component_50 = 50 * bleurt_rule_score
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

_persona = json.load(open(config.PERSONA_PROMPT_PATH, encoding="utf-8"))
_SIG = (_persona["hard_data"]["starters"]
        + _persona["hard_data"]["fillers"]
        + _persona["hard_data"]["endings"])
_SIG = [s.strip() for s in _SIG if s.strip()]


def _n_sentences(text: str) -> int:
    parts = re.split(r"[.!?。…\n]+", text.strip())
    return max(1, len([p for p in parts if p.strip()]))


def sentence_count_score(gen: str, gold: str) -> float:
    a, b = _n_sentences(gen), _n_sentences(gold)
    return 1.0 - min(1.0, abs(a - b) / max(b, 1))


def signature_phrase_score(gen: str) -> float:
    if not _SIG:
        return 0.0
    hit = sum(1 for s in _SIG if s in gen)
    # 1개 이상 등장하면 강하게, 그 이상은 점증 (상한 1.0)
    return min(1.0, hit / 3.0)


class Metrics:
    """무거운 모델(ko-sroberta, BLEURT)을 1회 로드해 재사용."""

    def __init__(self, use_bleurt: bool = True, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.embed = SentenceTransformer(config.EMBED_MODEL_NAME, device=device)
        self.bleurt = None
        if use_bleurt:
            self.bleurt = self._load_bleurt(device)

    def _load_bleurt(self, device):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            name = "Elron/bleurt-tiny-128"
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
            return {"tok": tok, "model": model, "torch": torch, "device": device}
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ BLEURT 로드 실패({e}) → semantic_similarity로 대체.")
            return None

    def semantic_similarity(self, gen: str, gold: str) -> float:
        va, vb = self.embed.encode([gen, gold])
        va = va / (np.linalg.norm(va) + 1e-8)
        vb = vb / (np.linalg.norm(vb) + 1e-8)
        return float(np.clip(va @ vb, 0.0, 1.0))

    def bleurt_raw(self, gen: str, gold: str) -> float | None:
        if not self.bleurt:
            return None
        t = self.bleurt
        with t["torch"].no_grad():
            inp = t["tok"]([gold], [gen], return_tensors="pt",
                           truncation=True, max_length=128).to(t["device"])
            score = t["model"](**inp).logits.squeeze().item()
        return float(score)

    @staticmethod
    def _bleurt_norm(raw: float) -> float:
        # BLEURT raw(대략 -1.5~1.0대)를 sigmoid로 [0,1] 매핑
        return float(1.0 / (1.0 + np.exp(-raw)))

    def score(self, gen: str, gold: str) -> dict:
        sc = sentence_count_score(gen, gold)
        sig = signature_phrase_score(gen)
        sem = self.semantic_similarity(gen, gold)
        heuristic_subavg = (sc + sig + sem) / 3.0

        raw = self.bleurt_raw(gen, gold)
        if raw is not None:
            bnorm = self._bleurt_norm(raw)
            source = "true_bleurt"
        else:
            bnorm = sem  # fallback
            source = "semantic_fallback"

        bleurt_rule = 0.4 * bnorm + 0.6 * heuristic_subavg
        return {
            "sentence_count_score": round(sc, 4),
            "signature_phrase_score": round(sig, 4),
            "semantic_similarity": round(sem, 4),
            "heuristic_subavg": round(heuristic_subavg, 4),
            "bleurt_raw": round(raw, 6) if raw is not None else None,
            "bleurt_norm": round(bnorm, 5),
            "bleurt_source": source,
            "bleurt_component_50": round(50 * bleurt_rule, 4),
        }
