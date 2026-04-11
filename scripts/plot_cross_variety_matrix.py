#!/usr/bin/env python3
"""
Plot a 3×3 cross-variety macro-F1 heatmap from JSON produced for Q2.2 / Q2.3.

Example:
  python3 scripts/plot_cross_variety_matrix.py \\
    --json reports/results/q2_3_lora_full_sarcasm.json \\
    --matrix-key mean_over_seeds \\
    --out reports/figures/q2_3_lora_macro_f1_heatmap.png \\
    --title "LoRA — cross-variety macro-F1 (Sarcasm)"

For Q2.2 RoBERTa, fill q2_2_roberta_crossvariety_sarcasm.json with numbers first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

VARIETIES = ["en-UK", "en-AU", "en-IN"]


def _load_matrix(data: dict, matrix_key: str) -> list[list[float | None]]:
    if "macro_f1" in data and isinstance(data["macro_f1"], dict):
        block = data["macro_f1"].get(matrix_key)
    else:
        block = data.get(matrix_key)
    if not isinstance(block, dict):
        raise ValueError(f"Could not find matrix under macro_f1['{matrix_key}']")

    rows: list[list[float | None]] = []
    for train_v in VARIETIES:
        row: list[float | None] = []
        inner = block.get(train_v, {})
        if not isinstance(inner, dict):
            inner = {}
        for test_v in VARIETIES:
            v = inner.get(test_v)
            row.append(None if v is None else float(v))
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot cross-variety macro-F1 heatmap")
    ap.add_argument("--json", required=True, help="Path to results JSON")
    ap.add_argument(
        "--matrix-key",
        default="mean_over_seeds",
        help="Key under macro_f1 (default: mean_over_seeds)",
    )
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--title", default="Cross-variety macro-F1", help="Plot title")
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)

    mat = _load_matrix(data, args.matrix_key)
    if any(any(x is None for x in r) for r in mat):
        print(
            "Error: matrix contains null/missing values. Fill the JSON first.",
            file=sys.stderr,
        )
        return 1

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    arr = np.array(mat, dtype=float)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(
        arr,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        xticklabels=VARIETIES,
        yticklabels=[f"Train {v}" for v in VARIETIES],
        ax=ax,
    )
    ax.set_xlabel("Test variety")
    ax.set_ylabel("Adapter / model trained on")
    ax.set_title(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close()
    print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
