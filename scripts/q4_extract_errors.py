"""Extract 10 misclassified test examples from the best LoRA adapter for Q4 (error analysis).

Picks a mixture of false positives and false negatives, distributed across the three
varieties when possible, and saves them to JSON in a layout matching the docx template
in reports/results/q4_error_analysis.md.

Usage (from repo root):

    python scripts/q4_extract_errors.py \
        --base-llm   facebook/opt-1.3b \
        --adapter    momofahmi/besstie-lora-en-uk-opt-1.3b \
        --task       Sarcasm \
        --n-errors   10 \
        --out        reports/results/q4_errors.json

You can pass --adapter multiple times to combine errors from per-variety adapters
(then the variety in `text → variety` field is used for routing).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATASET_ID = "surrey-nlp/BESSTIE-CW-26"
VARIETIES = ["en-UK", "en-AU", "en-IN"]
ADAPTER_BY_VARIETY = {
    "en-UK": "momofahmi/besstie-lora-en-uk-opt-1.3b",
    "en-AU": "momofahmi/besstie-lora-en-au-opt-1.3b",
    "en-IN": "momofahmi/besstie-lora-en-in-opt-1.3b",
}


def load_lora_model(base_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    base = AutoModelForSequenceClassification.from_pretrained(base_id, num_labels=2, dtype=dtype)
    base.config.pad_token_id = tok.pad_token_id

    peft = None
    for variety, adapter_id in ADAPTER_BY_VARIETY.items():
        if peft is None:
            peft = PeftModel.from_pretrained(base, adapter_id, adapter_name=variety)
        else:
            peft.load_adapter(adapter_id, adapter_name=variety)
    peft.eval().to(device)
    return tok, peft


@torch.no_grad()
def predict_one(tok, mdl, text: str, variety: str, device: str, max_length: int = 128) -> int:
    mdl.set_adapter(variety)
    enc = tok(text, return_tensors="pt", padding="max_length",
              truncation=True, max_length=max_length).to(device)
    logits = mdl(**enc).logits
    return int(logits.argmax(-1).item())


def pick_balanced_errors(errors: List[Dict], n: int) -> List[Dict]:
    """Try to spread the chosen errors over (variety, gold-label) buckets."""
    by_bucket: Dict[tuple, List[Dict]] = {}
    for e in errors:
        by_bucket.setdefault((e["variety"], e["gold"]), []).append(e)

    chosen: List[Dict] = []
    buckets = list(by_bucket.values())
    random.shuffle(buckets)
    while len(chosen) < n and any(buckets):
        for bucket in buckets:
            if not bucket or len(chosen) >= n:
                continue
            chosen.append(bucket.pop(random.randrange(len(bucket))))
    return chosen[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-llm", default="facebook/opt-1.3b")
    p.add_argument("--task", default="Sarcasm", choices=["Sarcasm", "Sentiment"])
    p.add_argument("--n-errors", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="reports/results/q4_errors.json")
    args = p.parse_args()

    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading dataset...")
    ds = load_dataset(DATASET_ID, split="test")

    print("Loading model + adapters...")
    tok, mdl = load_lora_model(args.base_llm, device)

    errors: List[Dict] = []
    for i, row in enumerate(ds):
        if i % 200 == 0:
            print(f"  {i}/{len(ds)}  | errors so far: {len(errors)}")
        gold = int(row[args.task])
        pred = predict_one(tok, mdl, row["text"], row["variety"], device)
        if pred != gold:
            errors.append({
                "idx": i,
                "text": row["text"],
                "variety": row["variety"],
                "source": row["source"],
                "gold": gold,
                "pred": pred,
                "task": args.task,
                "adapter": ADAPTER_BY_VARIETY[row["variety"]],
            })

    print(f"\nTotal misclassifications: {len(errors)}")
    chosen = pick_balanced_errors(errors, args.n_errors)
    print(f"Selected {len(chosen)} for Q4:")
    for e in chosen:
        print(f"  [{e['variety']}] gold={e['gold']} pred={e['pred']}  {e['text'][:100]!r}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "task": args.task,
            "base_llm": args.base_llm,
            "n_total_errors": len(errors),
            "n_selected": len(chosen),
            "examples": chosen,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
