"""Benchmark inference latency for the three model families used in the report.

Usage (from repo root):
    python scripts/benchmark_inference.py \
        --tfidf-vec  notebooks/models/tfidf/tfidf_vectorizer.pkl \
        --tfidf-clf  notebooks/models/LogisticRegression_sarcasm.pkl \
        --roberta    roberta-base \
        --base-llm   facebook/opt-1.3b \
        --lora       momofahmi/besstie-lora-en-uk-opt-1.3b \
        --out        reports/results/q5_2_efficiency.json

Any model flag can be omitted; the script will skip and report the rest.

Outputs
-------
- Mean / median / std latency (ms) for batch sizes [1, 32, 128] over N warmup + N timed runs.
- Param counts (trainable + total).
- A JSON dump suitable for the report.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

# Default sentences. Long enough to represent realistic inputs; mix of varieties.
SAMPLE_TEXTS = [
    "Absolute legend, parked his ute right across my driveway. Good onya, mate.",
    "Traditional friendly pub. Excellent beer.",
    "Coz we all have free internet.",
    "Yeah it blew out to 3x what it was budgeted for. Who wouldve thought giving people free cash to renovate their house would dry up resources for new builds...",
    "Cheerful fellow aren't you.",
    "I'll have to try one of these tomorrow morning, thanks!",
    "Where are Jhandvi's 1-2 bots who were claiming for her to be bigger than Kriti now?",
    "Best chats place in VD road from a long time.. too much crowd lately..",
]

BATCH_SIZES = [1, 32, 128]
N_WARMUP = 3
N_TIMED = 20


@dataclass
class LatencyRow:
    model: str
    device: str
    batch_size: int
    mean_ms: float
    median_ms: float
    std_ms: float
    p95_ms: float
    n_params_total: Optional[int] = None
    n_params_trainable: Optional[int] = None


def make_batch(n: int) -> List[str]:
    if n <= len(SAMPLE_TEXTS):
        return SAMPLE_TEXTS[:n]
    out = []
    while len(out) < n:
        out.extend(SAMPLE_TEXTS)
    return out[:n]


def time_fn(fn, n_warmup: int = N_WARMUP, n_timed: int = N_TIMED) -> dict:
    for _ in range(n_warmup):
        fn()
    samples_ms = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    samples_ms.sort()
    return {
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "std_ms": statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "p95_ms": samples_ms[int(0.95 * len(samples_ms))],
    }


def device_str() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
    except Exception:
        pass
    return "cpu"


# --------------------------------------------------------------------------- #
#  TF-IDF + Logistic Regression                                                #
# --------------------------------------------------------------------------- #
def benchmark_tfidf(vec_path: str, clf_path: str) -> List[LatencyRow]:
    import pickle
    import numpy as np  # noqa: F401  (sklearn pulls it in)

    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)

    n_total = sum(c.coef_.size + c.intercept_.size
                  for c in (clf.estimators_ if hasattr(clf, "estimators_") else [clf]))

    rows: List[LatencyRow] = []
    for bs in BATCH_SIZES:
        batch = make_batch(bs)

        def _one():
            X = vectorizer.transform(batch)
            _ = clf.predict(X)

        t = time_fn(_one)
        rows.append(LatencyRow(
            model="TF-IDF + LogReg",
            device="cpu",
            batch_size=bs,
            n_params_total=int(n_total),
            n_params_trainable=int(n_total),
            **t,
        ))
    return rows


# --------------------------------------------------------------------------- #
#  HuggingFace encoder (RoBERTa-base by default)                               #
# --------------------------------------------------------------------------- #
def benchmark_hf_encoder(model_id: str, device: str) -> List[LatencyRow]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2).to(device)
    mdl.eval()

    n_total = sum(p.numel() for p in mdl.parameters())
    n_train = sum(p.numel() for p in mdl.parameters() if p.requires_grad)

    rows: List[LatencyRow] = []
    for bs in BATCH_SIZES:
        batch = make_batch(bs)

        def _one():
            enc = tok(batch, return_tensors="pt", padding="max_length",
                      truncation=True, max_length=128).to(device)
            with torch.no_grad():
                _ = mdl(**enc).logits
            if device.startswith("cuda"):
                torch.cuda.synchronize()

        t = time_fn(_one)
        rows.append(LatencyRow(
            model=f"HF encoder ({model_id})",
            device=device,
            batch_size=bs,
            n_params_total=int(n_total),
            n_params_trainable=int(n_train),
            **t,
        ))
    return rows


# --------------------------------------------------------------------------- #
#  OPT-1.3B + LoRA adapter                                                     #
# --------------------------------------------------------------------------- #
def benchmark_lora(base_id: str, adapter_id: str, device: str) -> List[LatencyRow]:
    import os, sys
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _compat import ensure_peft_compat
    ensure_peft_compat()
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    base = AutoModelForSequenceClassification.from_pretrained(base_id, num_labels=2, dtype=dtype)
    base.config.pad_token_id = tok.pad_token_id

    mdl = PeftModel.from_pretrained(base, adapter_id, adapter_name="active")
    mdl = mdl.to(device).eval()

    n_total = sum(p.numel() for p in mdl.parameters())
    n_train = sum(p.numel() for p in mdl.parameters() if p.requires_grad)

    rows: List[LatencyRow] = []
    for bs in BATCH_SIZES:
        batch = make_batch(bs)

        def _one():
            enc = tok(batch, return_tensors="pt", padding="max_length",
                      truncation=True, max_length=128).to(device)
            with torch.no_grad():
                _ = mdl(**enc).logits
            if device.startswith("cuda"):
                torch.cuda.synchronize()

        t = time_fn(_one)
        rows.append(LatencyRow(
            model=f"LoRA on {base_id} (adapter: {adapter_id})",
            device=device,
            batch_size=bs,
            n_params_total=int(n_total),
            n_params_trainable=int(n_train),
            **t,
        ))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tfidf-vec", default=None, help="Path to TF-IDF vectorizer .pkl")
    p.add_argument("--tfidf-clf", default=None, help="Path to LogReg / SVM classifier .pkl")
    p.add_argument("--roberta", default=None, help="HF model id or local path for RoBERTa")
    p.add_argument("--base-llm", default="facebook/opt-1.3b", help="LoRA base LLM id")
    p.add_argument("--lora", default=None, help="LoRA adapter id (HF Hub) or local path")
    p.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu")
    p.add_argument("--out", default="reports/results/q5_2_efficiency.json")
    args = p.parse_args()

    device = args.device or ("cuda" if _cuda_ok() else "cpu")
    print(f"Device: {device_str()}\n")

    all_rows: List[LatencyRow] = []

    if args.tfidf_vec and args.tfidf_clf:
        print(f"[1/3] TF-IDF + LR  (vec={args.tfidf_vec}, clf={args.tfidf_clf})")
        try:
            all_rows += benchmark_tfidf(args.tfidf_vec, args.tfidf_clf)
        except Exception as e:
            print(f"  skipped: {e}")
    else:
        print("[1/3] TF-IDF: skipped (no --tfidf-vec / --tfidf-clf)")

    if args.roberta:
        print(f"[2/3] RoBERTa  ({args.roberta})")
        try:
            all_rows += benchmark_hf_encoder(args.roberta, device)
        except Exception as e:
            print(f"  skipped: {e}")
    else:
        print("[2/3] RoBERTa: skipped (no --roberta)")

    if args.lora:
        print(f"[3/3] LoRA on {args.base_llm}  (adapter={args.lora})")
        try:
            all_rows += benchmark_lora(args.base_llm, args.lora, device)
        except Exception as e:
            print(f"  skipped: {e}")
    else:
        print("[3/3] LoRA: skipped (no --lora)")

    # Print and save.
    print("\n" + _format_table(all_rows))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump([asdict(r) for r in all_rows], f, indent=2)
    print(f"\nWrote {args.out}")


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _format_table(rows: List[LatencyRow]) -> str:
    if not rows:
        return "(no rows)"
    header = f"{'Model':<55} {'Device':<10} {'BS':>4} {'mean':>8} {'median':>8} {'std':>8} {'p95':>8}  params"
    lines = [header, "-" * len(header)]
    for r in rows:
        params = ""
        if r.n_params_total:
            params = f"{r.n_params_total/1e6:.1f}M"
            if r.n_params_trainable and r.n_params_trainable != r.n_params_total:
                params += f" ({r.n_params_trainable/1e6:.2f}M trainable)"
        lines.append(
            f"{r.model[:55]:<55} {r.device[:10]:<10} {r.batch_size:>4} "
            f"{r.mean_ms:>7.2f}  {r.median_ms:>7.2f}  {r.std_ms:>7.2f}  {r.p95_ms:>7.2f}  {params}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
