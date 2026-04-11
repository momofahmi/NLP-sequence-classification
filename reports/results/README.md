# Results write-ups (for the PDF report)

Short analysis + tables live here so the group can copy into `docs/REPORT_TEMPLATE.md` without hunting notebooks.

| File | Section | Status |
|------|---------|--------|
| [q1_2_vocab_overlap.md](q1_2_vocab_overlap.md) | §1.2 | Filled from `local_run_summary.json` |
| [q2_1_baseline_tfidf.md](q2_1_baseline_tfidf.md) | §2.1 | Filled from `local_run_summary.json` |
| [q2_2_roberta_crossvariety_sarcasm.md](q2_2_roberta_crossvariety_sarcasm.md) | §2.2 | **Template** — paste FULL RoBERTa 3×3 grids from notebook 2.2 |
| [q2_2_roberta_crossvariety_sarcasm.json](q2_2_roberta_crossvariety_sarcasm.json) | — | **Template** (`null` → numbers) for plots / version control |
| [q2_3_lora_full_sarcasm.md](q2_3_lora_full_sarcasm.md) | §2.3 | Filled from your FULL LoRA run |
| [q2_3_lora_full_sarcasm.json](q2_3_lora_full_sarcasm.json) | — | Structured macro-F1 for scripts |

**Figures:** heatmaps for any 3×3 matrix can be generated with `scripts/plot_cross_variety_matrix.py` (see each `.md` file). LoRA mean matrix: `reports/figures/q2_3_lora_macro_f1_heatmap.png`.
