"""
Interactive RAG CLI demo.

  python rag/infer.py                       # interactive chat
  python rag/infer.py --once "질문"          # single turn
  python rag/infer.py --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from rag.chat import PersonaChat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.DEFAULT_RAG_MODEL)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--once", default=None, help="single query then exit")
    args = ap.parse_args()

    bot = PersonaChat(model_name=args.model, top_k=args.top_k)

    if args.once is not None:
        print(f"👤 사용자: {args.once}")
        print("😎 침착맨: ", end="", flush=True)
        for tok in bot.stream(args.once):
            print(tok, end="", flush=True)
        print()
        return

    print("=" * 50)
    print(f"⚡ 침착맨 RAG demo (model={args.model}, top_k={args.top_k})")
    print("quit: q / quit / exit")
    print("=" * 50)
    while True:
        try:
            user_input = input("\n👤 사용자: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"q", "quit", "exit"}:
            break
        if not user_input:
            continue
        print("😎 침착맨: ", end="", flush=True)
        for tok in bot.stream(user_input):
            print(tok, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
