"""Few-shot prompt evaluation for Q4.

Reads `reports/results/q4_errors.json` (output of q4_extract_errors.py), expects 4
of the 10 examples to have a non-empty `explanation` field, builds a 4-shot prompt
from those, then asks a generative LLM to predict the label of the remaining 6.

The generative LLM is configurable via --judge-model. Default is the same OPT-1.3B
in causal LM mode, but for stronger few-shot we recommend a small instruction-tuned
model the team already has access to (e.g. `meta-llama/Llama-3.2-3B-Instruct` or
`Qwen/Qwen2.5-1.5B-Instruct`).

Usage (from repo root):

    python scripts/q4_few_shot_eval.py \
        --in   reports/results/q4_errors.json \
        --judge-model  Qwen/Qwen2.5-1.5B-Instruct \
        --task Sarcasm \
        --out  reports/results/q4_fewshot_results.json

Workflow expected from Mohammad:

    1. Run q4_extract_errors.py → reports/results/q4_errors.json
    2. Open the JSON and write a one-paragraph `explanation` for any 4 of the 10
       examples. Leave the other 6 with `"explanation": ""`.
    3. Run this script. It will:
         - Take the 4 explained as the few-shot exemplars.
         - Test the 6 unexplained with the few-shot prompt.
         - Save before/after predictions to q4_fewshot_results.json.
    4. Paste the comparison into reports/results/q4_error_analysis.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List

LABELS_BY_TASK = {
    "Sarcasm": {1: "Sarcastic", 0: "Not Sarcastic"},
    "Sentiment": {1: "Positive", 0: "Negative"},
}


def build_prompt(exemplars: List[Dict], target_text: str, target_variety: str, task: str) -> str:
    name_by_label = LABELS_BY_TASK[task]
    instr = (
        f"You are a text classifier. For each input, output exactly one label "
        f"({' or '.join(name_by_label.values())}) followed by a one-sentence "
        f"explanation. Use the cultural and linguistic context of the indicated "
        f"English variety.\n\n"
    )
    shots = []
    for ex in exemplars:
        label_str = name_by_label[int(ex["gold"])]
        shots.append(
            f"Variety: {ex['variety']}\n"
            f"Text: {ex['text']}\n"
            f"Label: {label_str}\n"
            f"Explanation: {ex['explanation'].strip()}\n"
        )
    target = (
        f"Variety: {target_variety}\n"
        f"Text: {target_text}\n"
        f"Label:"
    )
    return instr + "\n".join(shots) + "\n" + target


def parse_label(generation: str, task: str) -> int:
    name_by_label = LABELS_BY_TASK[task]
    inv = {v.lower(): k for k, v in name_by_label.items()}
    g = generation.strip().lower()
    # try direct match in the first ~40 chars
    head = g[:60]
    for name, lbl in inv.items():
        if name in head:
            return lbl
    # fall back to longest match anywhere
    matches = [(g.find(n), lbl) for n, lbl in inv.items() if n in g]
    if matches:
        matches.sort()
        return matches[0][1]
    return -1   # unparsable


def run_generative(judge_model: str, prompts: List[str], max_new_tokens: int = 64) -> List[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(judge_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(judge_model, dtype=dtype).to(device).eval()

    out: List[str] = []
    for p in prompts:
        enc = tok(p, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            gen = mdl.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(gen[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        out.append(text)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default="reports/results/q4_errors.json")
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--task", default=None,
                   help="Sarcasm | Sentiment. Defaults to whatever is recorded in the input JSON.")
    p.add_argument("--out", default="reports/results/q4_fewshot_results.json")
    args = p.parse_args()

    with open(args.in_path) as f:
        bundle = json.load(f)
    task = args.task or bundle.get("task", "Sarcasm")
    examples = bundle["examples"]

    explained = [e for e in examples if e.get("explanation", "").strip()]
    unexplained = [e for e in examples if not e.get("explanation", "").strip()]

    if len(explained) < 4:
        print(f"Need 4 explained examples; found {len(explained)}. "
              f"Add `explanation` strings to {args.in_path} and re-run.", file=sys.stderr)
        sys.exit(1)
    if len(unexplained) < 6:
        print(f"Warning: only {len(unexplained)} unexplained examples found; "
              f"the brief asks for 6.", file=sys.stderr)

    exemplars = explained[:4]
    targets = unexplained[:6]

    prompts = [build_prompt(exemplars, t["text"], t["variety"], task) for t in targets]

    print(f"Running few-shot eval on {len(prompts)} examples with judge={args.judge_model} ...")
    generations = run_generative(args.judge_model, prompts)

    rows = []
    for tgt, prompt, gen in zip(targets, prompts, generations):
        new_pred = parse_label(gen, task)
        rows.append({
            **tgt,
            "before_pred": int(tgt["pred"]),
            "after_pred": int(new_pred),
            "raw_generation": gen.strip(),
            "prompt_used": prompt,
            "improved": int(new_pred == int(tgt["gold"])),
        })

    n_correct_after = sum(r["improved"] for r in rows)
    n_correct_before = sum(int(r["before_pred"] == int(r["gold"])) for r in rows)

    summary = {
        "task": task,
        "judge_model": args.judge_model,
        "exemplars": exemplars,
        "n_targets": len(rows),
        "n_correct_before": n_correct_before,
        "n_correct_after": n_correct_after,
        "delta": n_correct_after - n_correct_before,
        "results": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nBefore (LoRA):  {n_correct_before}/{len(rows)} correct")
    print(f"After  (4-shot): {n_correct_after}/{len(rows)} correct  (Δ={summary['delta']:+d})")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
