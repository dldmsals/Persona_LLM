"""
Qwen3 QLoRA SFT 학습 (trl SFTTrainer).

- 4-bit nf4 double-quant + bf16, LoRA r=16
- GPU 선택은 SLURM 에 위임 (CUDA_VISIBLE_DEVICES 하드코딩 금지)

  python finetune/train_qlora.py --base Qwen/Qwen3-0.6B --epochs 1   # sanity
  python finetune/train_qlora.py --base Qwen/Qwen3-4B   --epochs 3   # 본 학습
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "sft_data.jsonl"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=0, help=">0 이면 일부만(빠른 검증)")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    ds = load_dataset("json", data_files=args.data, split="train")
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    def fmt(ex):
        return {"text": tok.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)}
    ds = ds.map(fmt, remove_columns=ds.column_names)

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=args.max_len,
        packing=False,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         peft_config=lora, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"✅ 학습 완료. 어댑터 저장: {args.out}")


if __name__ == "__main__":
    main()
