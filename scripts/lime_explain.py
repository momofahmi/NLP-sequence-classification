"""LIME explanations for misclassified sarcasm examples.

Produces per-example LIME explanations (HTML + token-importance bar PNG) for any
of the three modelling families. Intended to be referenced from §2.2 / §4 of the
report as evidence that the models latch onto specific lexical cues (and what
those cues are).

Usage (from repo root):

    # Default: explain the 4 'explained' examples in q4_errors.json with the
    # OPT-1.3B + LoRA model (per-variety adapter, picked automatically).
    python scripts/lime_explain.py \
        --model      lora \
        --in         reports/results/q4_errors.json \
        --out-dir    reports/figures/lime/

    # Or explain the same examples with the TF-IDF + LR baseline (CPU, fast).
    python scripts/lime_explain.py \
        --model      tfidf \
        --tfidf-vec  notebooks/models/tfidf/tfidf_vectorizer.pkl \
        --tfidf-clf  notebooks/models/LogisticRegression_sarcasm.pkl \
        --in         reports/results/q4_errors.json \
        --out-dir    reports/figures/lime/

    # Or explain a single arbitrary sentence:
    python scripts/lime_explain.py \
        --model    lora \
        --variety  en-AU \
        --text     "Absolute legend, parked his ute right across my driveway. Good onya, mate." \
        --out-dir  reports/figures/lime/

Dependencies:  pip install lime
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

LABELS_BY_TASK = {
    "Sarcasm": ["Not Sarcastic", "Sarcastic"],
    "Sentiment": ["Negative", "Positive"],
}

ADAPTER_BY_VARIETY = {
    "en-UK": "momofahmi/besstie-lora-en-uk-opt-1.3b",
    "en-AU": "momofahmi/besstie-lora-en-au-opt-1.3b",
    "en-IN": "momofahmi/besstie-lora-en-in-opt-1.3b",
}


def make_lora_predict_fn(base_id: str, variety: str):
    """Return a `predict_proba` callable that takes List[str] and returns N x 2."""
    import numpy as np
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _compat import ensure_peft_compat
    ensure_peft_compat()
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(base_id, num_labels=2, dtype=dtype)
    base.config.pad_token_id = tok.pad_token_id

    mdl = PeftModel.from_pretrained(base, ADAPTER_BY_VARIETY[variety], adapter_name=variety)
    mdl.set_adapter(variety)
    mdl.eval().to(device)

    @torch.no_grad()
    def predict_proba(texts: List[str]) -> "np.ndarray":
        enc = tok(list(texts), return_tensors="pt", padding="max_length",
                  truncation=True, max_length=128).to(device)
        logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    return predict_proba


def make_roberta_predict_fn(model_id: str):
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2).to(device).eval()

    @torch.no_grad()
    def predict_proba(texts: List[str]) -> "np.ndarray":
        enc = tok(list(texts), return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to(device)
        logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    return predict_proba


def make_tfidf_predict_fn(vec_path: str, clf_path: str):
    import pickle
    import numpy as np

    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)

    def predict_proba(texts: List[str]) -> "np.ndarray":
        X = vectorizer.transform(list(texts))
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X)
        # SVMs without probability=True: convert decision function to softmax
        df = clf.decision_function(X)
        if df.ndim == 1:
            df = np.stack([-df, df], axis=1)
        e = np.exp(df - df.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    return predict_proba


def explain_one(predict_proba, text: str, label_idx: int, class_names, num_features: int = 10):
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=class_names)
    return explainer.explain_instance(
        text, predict_proba, num_features=num_features, labels=[label_idx]
    )


def save_explanation(exp, label_idx: int, out_stem: str, title: str):
    import matplotlib.pyplot as plt

    html_path = out_stem + ".html"
    png_path = out_stem + ".png"

    exp.save_to_file(html_path)

    fig = exp.as_pyplot_figure(label=label_idx)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return html_path, png_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["lora", "roberta", "tfidf"], default="lora")
    p.add_argument("--task", default="Sarcasm", choices=list(LABELS_BY_TASK.keys()))
    p.add_argument("--base-llm", default="facebook/opt-1.3b")
    p.add_argument("--roberta", default="roberta-base")
    p.add_argument("--tfidf-vec", default="notebooks/models/tfidf/tfidf_vectorizer.pkl")
    p.add_argument("--tfidf-clf", default="notebooks/models/LogisticRegression_sarcasm.pkl")
    p.add_argument("--in", dest="in_path", default=None,
                   help="Path to q4_errors.json. If set, explains the 4 examples that have non-empty `explanation`.")
    p.add_argument("--text", default=None, help="Single ad-hoc sentence to explain.")
    p.add_argument("--variety", default="en-UK", choices=list(ADAPTER_BY_VARIETY))
    p.add_argument("--num-features", type=int, default=10)
    p.add_argument("--out-dir", default="reports/figures/lime")
    args = p.parse_args()

    class_names = LABELS_BY_TASK[args.task]
    os.makedirs(args.out_dir, exist_ok=True)

    # Decide what to explain.
    examples = []
    if args.text:
        examples.append({"text": args.text, "variety": args.variety,
                         "gold": -1, "pred": -1, "explanation": "(ad-hoc)"})
    elif args.in_path:
        with open(args.in_path) as f:
            bundle = json.load(f)
        for ex in bundle.get("examples", []):
            if ex.get("explanation", "").strip():
                examples.append(ex)
        if not examples:
            print(f"No explained examples found in {args.in_path}. "
                  f"Add an `explanation` field to 4 of the entries first.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Need either --text or --in <q4_errors.json>", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.model} model...")
    # We build one predict_fn per (model_family, variety). For LoRA, we may need to
    # rebuild per example because the adapter changes. For others, build once.
    cached_predict = {}
    if args.model == "roberta":
        cached_predict["_"] = make_roberta_predict_fn(args.roberta)
    elif args.model == "tfidf":
        cached_predict["_"] = make_tfidf_predict_fn(args.tfidf_vec, args.tfidf_clf)

    summary_rows = []
    for i, ex in enumerate(examples):
        text = ex["text"]
        variety = ex["variety"]
        gold = int(ex["gold"]) if ex["gold"] != -1 else None
        pred = int(ex["pred"]) if ex["pred"] != -1 else None

        if args.model == "lora":
            if variety not in cached_predict:
                cached_predict[variety] = make_lora_predict_fn(args.base_llm, variety)
            predict_fn = cached_predict[variety]
            model_label = f"OPT-1.3B+LoRA[{variety}]"
        elif args.model == "roberta":
            predict_fn = cached_predict["_"]
            model_label = f"RoBERTa-base"
        else:
            predict_fn = cached_predict["_"]
            model_label = "TF-IDF + LR"

        # Use the model's prediction as the LIME label-of-interest if pred is known,
        # otherwise the gold label, otherwise label 1 (Sarcastic / Positive).
        label_idx = pred if pred is not None and pred >= 0 else (gold if gold is not None else 1)

        proba = predict_fn([text])[0]
        title = (f"{model_label} | {variety} | "
                 f"gold={class_names[gold] if gold is not None else 'n/a'} | "
                 f"pred={class_names[label_idx]} ({proba[label_idx]:.0%})")

        print(f"[{i+1}/{len(examples)}] explaining: {text[:80]!r} ...")
        exp = explain_one(predict_fn, text, label_idx, class_names, args.num_features)
        out_stem = os.path.join(args.out_dir, f"lime_{args.model}_{variety}_{i+1:02d}")
        html_path, png_path = save_explanation(exp, label_idx, out_stem, title)

        word_weights = exp.as_list(label=label_idx)
        summary_rows.append({
            "i": i + 1,
            "model": model_label,
            "variety": variety,
            "text": text,
            "gold": gold,
            "pred": label_idx,
            "proba_pred": float(proba[label_idx]),
            "top_features": word_weights,
            "html": html_path,
            "png": png_path,
        })
        print(f"  -> {png_path}")

    # Save aggregate JSON.
    out_json = os.path.join(args.out_dir, f"lime_{args.model}_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    print(f"\nWrote summary: {out_json}")


if __name__ == "__main__":
    main()
