# Results write-ups (for the PDF report)

Short analysis + tables live here so the group can copy into `docs/report_template.md` without hunting notebooks.

| File | Section | Status |
|------|---------|--------|
| [q1_2_vocab_overlap.md](q1_2_vocab_overlap.md) | §1.2 | Filled from `local_run_summary.json` |
| [q2_1_baseline_tfidf.md](q2_1_baseline_tfidf.md) | §2.1 | Filled from `local_run_summary.json` |
| [q2_2_roberta_crossvariety_sarcasm.md](q2_2_roberta_crossvariety_sarcasm.md) | §2.2 | Filled — variety-only 3×3 + `inner_pool` / `all`; heatmap: `../figures/q2_2_roberta_macro_f1_heatmap.png` |
| [q2_2_roberta_crossvariety_sarcasm.json](q2_2_roberta_crossvariety_sarcasm.json) | — | Structured macro-F1 + extra pools |
| [q2_3_lora_full_sarcasm.md](q2_3_lora_full_sarcasm.md) | §2.3 | Filled — LoRA FULL run |
| [q2_3_lora_full_sarcasm.json](q2_3_lora_full_sarcasm.json) | — | Structured macro-F1 for scripts |

**Figures:** `python3 scripts/plot_cross_variety_matrix.py --help` — LoRA: `../figures/q2_3_lora_macro_f1_heatmap.png`; RoBERTa: `../figures/q2_2_roberta_macro_f1_heatmap.png`.
