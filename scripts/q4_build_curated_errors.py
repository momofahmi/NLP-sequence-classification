"""Build reports/results/q4_errors.json from the 10 examples discussed in the report.

Section 4 of the report identifies 10 specific en-AU test examples that the
OPT-1.3B + en-AU LoRA adapter misclassifies, and provides written explanations
for 4 of them. This script reproduces that file by:

    1. Loading the BESSTIE-CW-26 test split.
    2. Filtering to en-AU only (so positional indices match the report).
    3. Pulling the 10 specific en-AU positions discussed in section 4.
    4. Adding the 4 hand-written explanations from the report.

Run once after cloning the repo (no model required, just the dataset):

    python scripts/q4_build_curated_errors.py

Will write `reports/results/q4_errors.json` and is safe to re-run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASET_ID = "surrey-nlp/BESSTIE-CW-26"

ADAPTER_BY_VARIETY = {
    "en-UK": "momofahmi/besstie-lora-en-uk-opt-1.3b",
    "en-AU": "momofahmi/besstie-lora-en-au-opt-1.3b",
    "en-IN": "momofahmi/besstie-lora-en-in-opt-1.3b",
}

# 10 examples from the report (en-AU test split, indexed within the en-AU subset).
# `gold` and `pred` reflect the OPT-1.3B + en-AU adapter behaviour reported in §4.
# 4 of them carry a written explanation; the other 6 are the held-out targets
# for the few-shot evaluation.
REPORT_EXAMPLES = [
    {
        "idx_en_au": 142, "gold": 1, "pred": 0,
        "explanation": (
            "Sarcastic. The first clause 'It's great' is positive on the surface but "
            "is immediately undercut by 'barely any customers and the cinemas are "
            "always empty', which makes the positive evaluation absurd. The cue is "
            "an inter-clause incongruity rather than any single token: a model that "
            "reads 'great' on its own (as TF-IDF or a non-contextual model would) "
            "predicts not sarcastic. OPT-1.3B + LoRA still falls into this trap "
            "because the adapter is small and the training set has few en-AU "
            "examples of this construction."
        ),
    },
    {
        "idx_en_au": 302, "gold": 1, "pred": 0,
        "explanation": (
            "Sarcastic. The setup ('we recently spent a fair bit of money on a very "
            "nice dining table and sturdy comfortable dining chairs') is contradicted "
            "by the punchline ('we eat on the lounge watching our stories'). 'So to "
            "answer your question' is a discourse marker that signals the punchline. "
            "The contradiction lives across multiple sentences, which is hard for a "
            "model trained on 128-token windows and short Reddit comments. The "
            "Australian colloquial 'lounge' (= sofa) does not help either."
        ),
    },
    {
        "idx_en_au": 508, "gold": 0, "pred": 1,
        "explanation": (
            "Not sarcastic. The speaker is making a sincere first-person complaint "
            "about a 50% rent increase since 2020 and explicitly denies being a "
            "shill ('Not a shill mate'). Reddit-style colloquial register and "
            "emphatic punctuation can fool a model that has learned 'angry tone == "
            "sarcasm', but here the angry tone matches the literal grievance, not "
            "an ironic one. The Australian 'mate' adds register noise without "
            "changing the literal meaning."
        ),
    },
    {
        "idx_en_au": 618, "gold": 0, "pred": 1,
        "explanation": (
            "Not sarcastic. The text is purely factual: it states that 5 weeks of "
            "annual leave is standard for shift workers and the other 4 weeks are "
            "usually ADOs (Accrued Days Off, an Australian-specific term). There is "
            "no contradiction, no exaggeration, and no evaluative language. The "
            "matter-of-fact register and the unfamiliar acronym are easy to "
            "misread as deadpan irony, which is what the model does here."
        ),
    },
    {"idx_en_au": 264, "gold": 1, "pred": 0, "explanation": ""},
    {"idx_en_au": 523, "gold": 1, "pred": 0, "explanation": ""},
    {"idx_en_au": 657, "gold": 1, "pred": 0, "explanation": ""},
    {"idx_en_au": 256, "gold": 0, "pred": 1, "explanation": ""},
    {"idx_en_au": 395, "gold": 0, "pred": 1, "explanation": ""},
    {"idx_en_au": 492, "gold": 0, "pred": 1, "explanation": ""},
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="reports/results/q4_errors.json")
    args = p.parse_args()

    from datasets import load_dataset

    print(f"Loading {DATASET_ID} test split...")
    ds = load_dataset(DATASET_ID, split="test")
    en_au = [row for row in ds if row["variety"] == "en-AU"]
    print(f"en-AU test split: {len(en_au)} examples")

    out_examples = []
    for entry in REPORT_EXAMPLES:
        i = entry["idx_en_au"]
        if i >= len(en_au):
            print(f"  skipped idx {i}: out of range")
            continue
        row = en_au[i]
        out_examples.append({
            "idx": i,
            "text": row["text"],
            "variety": row["variety"],
            "source": row["source"],
            "gold": int(entry["gold"]),
            "pred": int(entry["pred"]),
            "task": "Sarcasm",
            "adapter": ADAPTER_BY_VARIETY[row["variety"]],
            "explanation": entry["explanation"],
        })

    n_explained = sum(1 for e in out_examples if e["explanation"].strip())

    payload = {
        "task": "Sarcasm",
        "base_llm": "facebook/opt-1.3b",
        "n_total_errors": 131,
        "n_selected": len(out_examples),
        "examples": out_examples,
        "_provenance": (
            "Built by scripts/q4_build_curated_errors.py from the 10 en-AU test "
            "examples discussed in section 4 of the report. Indices are positions "
            "in the en-AU-only filter of surrey-nlp/BESSTIE-CW-26 test split."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"Wrote {out_path}: {len(out_examples)} examples, "
          f"{n_explained} with explanations.")


if __name__ == "__main__":
    main()
