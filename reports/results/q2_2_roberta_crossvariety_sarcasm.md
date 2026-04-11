# Q2.2 — RoBERTa cross-variety (FULL), Sarcasm

## Status

**Full cross-variety macro-F1 matrices from your GPU run are not stored in this repo yet** (only a small **smoke** entry exists in `local_run_summary.json` under `roberta_smoke_sarcasm_train_en-UK`). This file is the **same style** as `q2_3_lora_full_sarcasm.md` so you can paste numbers in one place for the PDF.

### Why `q2_2_roberta_crossvariety_sarcasm.json` still has `null`

The JSON was added as a **scaffold** so the same workflow as LoRA (tables + `plot_cross_variety_matrix.py`) works once numbers are in git. **RoBERTa FULL outputs only existed in your Colab / chat messages** until someone copies the 3×3 grids into this file — unlike LoRA, we never committed those digits to the repo. **`null` means “paste your notebook results here,”** not “RoBERTa failed.” After you fill the matrix, set `"status": "filled"` and run the plot command below.

### What to paste here

From notebook `2.2_RoBERTa_CrossVariety_Joel_Fiyin.ipynb` after a **FULL** run (`DEMO_MODE=0`, seeds 42 & 123):

1. For **each seed**, the **3×3 macro-F1** grid (train variety × test variety).
2. Optionally, the **mean over seeds** (cell-by-cell average), like the LoRA doc.

If you already exported **figures** under `reports/figures/` (e.g. `q2_2 (RoBERTa) Cross-Variety Evaluation Matrix - Macro-F1.png`), the **heatmap in the figure is sufficient** for the report — but **tables + short prose** here make writing §2.2 faster and keep numbers version-controlled.

---

## Template — macro-F1 (replace `TODO` with your FULL run)

### Seed 42

| Trained on ↓ / Test → | en-UK | en-AU | en-IN |
|----------------------|------:|------:|------:|
| **en-UK** | TODO | TODO | TODO |
| **en-AU** | TODO | TODO | TODO |
| **en-IN** | TODO | TODO | TODO |

### Seed 123

| Trained on ↓ / Test → | en-UK | en-AU | en-IN |
|----------------------|------:|------:|------:|
| **en-UK** | TODO | TODO | TODO |
| **en-AU** | TODO | TODO | TODO |
| **en-IN** | TODO | TODO | TODO |

### Mean over seeds

| Trained on ↓ / Test → | en-UK | en-AU | en-IN |
|----------------------|------:|------:|------:|
| **en-UK** | TODO | TODO | TODO |
| **en-AU** | TODO | TODO | TODO |
| **en-IN** | TODO | TODO | TODO |

---

## Experimental setup (fill from notebook)

| Setting | Value |
|--------|--------|
| Model | `roberta-base` (or as run) |
| Mode | FULL (`DEMO_MODE=0`) |
| Epochs | 3 |
| Seeds | 42, 123 |
| Task | Sarcasm |
| Metric | Macro-F1 |

---

## Figure

After filling `q2_2_roberta_crossvariety_sarcasm.json` (see template) or exporting a CSV, generate a heatmap:

```bash
python3 scripts/plot_cross_variety_matrix.py \
  --json reports/results/q2_2_roberta_crossvariety_sarcasm.json \
  --matrix-key mean_over_seeds \
  --out reports/figures/q2_2_roberta_macro_f1_heatmap.png \
  --title "RoBERTa — cross-variety macro-F1 (Sarcasm)"
```

---

## Smoke reference (not for final report)

From `local_run_summary.json` — **tiny subset / smoke** only:

- Train en-UK, test en-UK / en-AU / en-IN macro-F1 ≈ **0.48 / 0.40 / 0.48** (illustrative only).
