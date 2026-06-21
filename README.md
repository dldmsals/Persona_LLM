# Persona LLM — RAG vs. QLoRA for Korean Character Replication

Replicating a specific Korean persona (the internet broadcaster **Chimchakman**) from
limited, noisy speech data, and a **methodology comparison**: is it better to *retrieve*
the persona's utterances (RAG) or to *bake* them into the weights (QLoRA fine-tuning)?

The two approaches use the **same persona QA data** — RAG retrieves it at inference time,
QLoRA internalizes it during training — which makes the comparison apples-to-apples.

## Pipeline
```
YouTube speech ──STT(WhisperX)──► utterance corpus
        │
        ├─ persona profile : Kiwi tic stats + Gemini rhetoric  → data/chim_persona_prompt.json
        ├─ QA generation   : claims→QA + persona-conditioned QA → data/chim_qa_*, chim_absurd_qa_*
        ▼
  ┌──────────────┬─────────────────────────┬──────────────────────┐
  │ RAG INDEX     │ SFT / QLoRA             │ TEST (held-out)      │
  │ 11,231 QA     │ 4,297 messages          │ 100 from chim_qa_1   │
  │ (embed Q)     │ (excl. chim_qa_1)       │ (not in index/train) │
  └──────────────┴─────────────────────────┴──────────────────────┘
```

## Setup
```bash
conda create -n persona python=3.10 -y && conda activate persona
pip install -r requirements.txt
cp .env.example .env        # add your OPENROUTER_API_KEY
```

## Run (no GPU needed)
```bash
# 1) Build the RAG index (CPU, ~1–2 min)
python rag/build_index.py

# 2) Interactive demo
python rag/infer.py                          # chat
python rag/infer.py --once "탕수육은 부먹이지?"  # single turn

# 3) Evaluate (held-out 100 items)
python eval/build_testset.py --n 100
python eval/run_eval.py --systems rag --model google/gemini-2.5-flash
python eval/run_eval.py --systems rag --model openai/gpt-4o-mini   # multi-provider

# 4) Analysis
python eval/aggregate.py        # comparison table + chart
python eval/significance.py     # 95% CIs + paired bootstrap
python eval/mechanism.py        # output-statistics diagnosis
python eval/topk_sweep.py       # RAG top-k curve
```

## Fine-tuning (QLoRA, GPU / SLURM)
```bash
python finetune/prepare_data.py                       # → finetune/sft_data.jsonl
sbatch finetune/train.sbatch                          # Qwen3-0.6B (default)
BASE=Qwen/Qwen3-4B EPOCHS=3 sbatch finetune/train.sbatch
# generate answers with the trained adapter, then score on CPU:
sbatch finetune/gen.sbatch                            # → answers_ft_*.jsonl
python eval/score_file.py --answers <answers.jsonl> --name ft_Qwen3-4B
```

## Evaluation
`Total(100) = BLEURT/Rule(50) + LLM-as-Judge(50)`
- **BLEURT/Rule**: BLEURT + sentence-count + signature-phrase + semantic similarity
- **LLM-as-Judge**: content_quality / persona_alignment / rhetorical_alignment (0–1), gpt-4o-mini via OpenRouter, temperature 0

## Key findings (n=100)
| Condition (Qwen3-4B) | Total | vs base | sig. |
|---|---|---|---|
| RAG (base + retrieval) | 61.9 | +4.6 | p<0.001 |
| base (prompt only) | 57.2 | — | — |
| RAG+FT | 54.8 | −2.5 | n.s. |
| FT (QLoRA) | 52.8 | −4.4 | p<0.001 |

- **Retrieval (RAG) is the dominant factor**; on a strong API model (gemini ≈ gpt-4o-mini, p=0.35) retrieval adds little.
- **Fine-tuning on noisy data underperforms** — it strips persona signature phrases (≈1.98 → 0.67 per answer) while diversity stays high.
- More SFT data helps monotonically (250→4297: 42→48) but plateaus below base/RAG → **data quality is the deciding factor**.

## Layout
```
config.py            paths, .env, API keys
rag/                 retriever · chat · infer · build_index
persona/             feature_extract · generate_qa
finetune/            prepare_data · train_qlora · gen_ft_answers · *.sbatch
eval/                build_testset · metrics · judge · run_eval · score_file
                     significance · mechanism · topk_sweep · aggregate
tools/               make_analysis · make_sweeps (figures)
data/                QA datasets, persona profile, claims
results/             scores, figures
```
