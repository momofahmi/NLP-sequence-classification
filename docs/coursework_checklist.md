# Coursework checklist

> Updated: 30 Apr 2026. Submission deadline: **Wed 6 May 2026, 16:00 UK time** — 6 days.

Tick items by editing this file (`[ ]` → `[x]`). The report-ready prose blocks for each section are under `reports/results/`. Paste them into the docx (Yusrah's Google Doc → final PDF named `report_PG##.pdf`).

> **Master document**: see `docs/report_outline.md` for the canonical section structure, page-budget table, and Google Docs formatting guide. Apply that outline to the Google Doc before pasting prose blocks.

---

## §1 — Data & EDA (15 marks, max 4 pages)

- [x] §1.1 EDA — written by Yusrah in the docx (figures `q1_1_*`, slang analysis, POS, sarcasm-sentiment correlation 97.52%)
- [x] §1.2 Vocabulary analysis — **prose ready** in `reports/results/q1_2_vocab_overlap.md`; figure `notebooks/reports/figures/vocabulary_similarity_heatmap.png`
- [ ] **PDF only:** paste §1.2 into the docx (currently has placeholder bullets)

---

## §2 — Experimentation (40 marks, max 6 pages)

### §2.1 — Baseline / PTLM gap (10 marks)
- [x] TF-IDF + LogReg (Yusrah) — written
- [x] SVM + TF-IDF (Omar) — code merged on `main`, results integrated in §2.1
- [x] RoBERTa-base baseline comparison (Yusrah) — written

### §2.2 — RoBERTa cross-variety (15 marks)
- [x] Notebook + WeightedTrainer (Joel) — on `origin/Joel`
- [x] Cross-variety matrix figures (`weighted_figures/cross_variety_matrix.png`, `confusion_matrix_best.png`) — on `origin/Joel`
- [x] Per-condition JSON results (`weighted_results/{uk_only,au_only,in_only,inner_pool,all}.json`) — on `origin/Joel`
- [ ] **Blocker:** **merge `origin/Joel` into `main`** — currently nested under `NLP-sequence-classification/`, must be flattened
- [ ] **PDF only:** fill the placeholders in §2.2 ("MACRO F1 COMPARISON NEEDED", "Need to add about RoBERTa here")
- [x] Earlier RoBERTa write-up & heatmap on `fiyin/model-pipeline` (`reports/results/q2_2_roberta_crossvariety_sarcasm.md` + `.json`, `reports/figures/q2_2_roberta_macro_f1_heatmap.png`) — can be salvaged into the final doc

### §2.3 — LoRA adapters (15 marks)
- [x] Notebooks: `lora_notebook_opt1.3B.ipynb` (canonical), `_llama1B`, `_llama3.2B_v2` (Mohamed)
- [x] Adapters trained: 6 (3 varieties × 2 seeds) on OPT-1.3B, all on HuggingFace Hub (`momofahmi/besstie-lora-en-{uk,au,in}-opt-1.3b`)
- [x] Cross-variety matrix in docx (Mohamed) — Tables 2 (Macro-F1 ± std) and 3 (Sarcasm-class F1)
- [x] Result figures: `results/opt1.3B/{ablation,confusion_matrices,cross_variety_matrices,training_curves}.png`
- [ ] **PDF only:** decide whether to keep the LLaMA-1B / LLaMA-3.2-3B comparison in the main body or move to an appendix / single sentence

---

## §3 — Evaluation (15 marks, max 5 pages)

- [x] LR + TF-IDF: Macro-F1 + per-class precision/recall + sarcasm CM analysis (Yusrah) — written
- [ ] RoBERTa: per-class precision/recall + best-model confusion matrix in docx — **uses Joel's outputs**
- [ ] LoRA: per-class precision/recall + best-model confusion matrix in docx — figure exists at `results/opt1.3B/confusion_matrices.png`

---

## §4 — Sarcasm Explanation & Error Analysis (10 marks, max 4 pages)

- [x] Scripts in place:
  - `scripts/q4_extract_errors.py` — pulls 10 misclassifications, balanced across (variety, gold-label)
  - `scripts/q4_few_shot_eval.py` — builds 4-shot prompt from explanations, tests remaining 6 with a configurable judge LLM
- [x] **Template ready** in `reports/results/q4_error_analysis.md`
- [ ] **Mohammad to do:**
  1. Run `scripts/q4_extract_errors.py`
  2. Write 4 explanations directly in `reports/results/q4_errors.json` (the `explanation` field on 4 of the 10 examples)
  3. Run `scripts/q4_few_shot_eval.py`
  4. Paste filled template into the docx

---

## §5 — Deployment (20 marks)

### §5.1 — Endpoint (15 marks, max 5 pages)
- [x] Gradio app `app/app.py` (Mohamed) — single-input + batch-compare tabs, hot-swaps adapters via `peft_model.set_adapter(variety)`
- [x] **Write-up ready** in `reports/results/q5_1_deployment.md`
- [ ] **Mohamed to do:** capture 3 screenshots flagged in the doc (`[FIGURE 5.1.1/2/3]`) on a live run; add to docx

### §5.2 — Efficiency (5 marks, max 1 page)
- [x] Benchmark script `scripts/benchmark_inference.py` — TF-IDF / RoBERTa / OPT-1.3B+LoRA at BS={1,32,128}
- [x] **Write-up template** in `reports/results/q5_2_efficiency.md`
- [ ] **One person to do:** run the script once on the team's reference hardware, drop numbers into Table 5.2.1

---

## Submission hygiene

- [ ] **Confirm group code (PG##)** — needed for `report_PG##.pdf` and ZIP filename
- [ ] Title page + declaration of originality (template on My Surrey)
- [ ] References list (BESSTIE paper, RoBERTa, LoRA, QLoRA, Plank 2022, Abercrombie & Hovy 2016, Skalicky & Crossley 2018, eWAVE, etc.)
- [ ] **Page-budget check** — current docx draft fits, but §2.3 is detailed; trim if total content exceeds 25 pages
- [ ] Build clean code ZIP: `./scripts/build_submission_zip.sh PG##` (excludes checkpoints, datasets, large `.arrow` and `.npz` files)
- [ ] Verify the canonical notebooks rerun top-to-bottom on a fresh Colab (`NLP_EDA.ipynb`, `SVM_TFIDF.ipynb`, Joel's `task-2.2.ipynb`, `lora_notebook_opt1.3B.ipynb`)
- [ ] Top-level `README.md` lists which notebook/script produces each figure cited in the report

---

## Branch state (30 Apr)

| Branch | Status |
|---|---|
| `main` | Hub. Has Mohamed's LoRA + Gradio app + Yusrah/Omar EDA/SVM. **Missing**: Joel's RoBERTa work, this branch's docs/scripts. |
| `origin/Joel` | Has RoBERTa notebook + weighted/non-weighted JSONs + figures, in a nested folder. **Needs to be merged into `main` (with the folder flattened).** |
| `origin/mohamedfahmi` | Already merged into `main`. |
| `origin/YusrahS`, `origin/YusrahS-EDA` | Content already on `main`. |
| `fiyin/model-pipeline` | Carries: `docs/`, `reports/results/`, `reports/figures/q2_2_roberta_*`, `scripts/sanitize_notebook.py`, `scripts/plot_cross_variety_matrix.py`, **all the new section drafts and Q4/Q5 scripts added 30 Apr**. Cherry-pick into `main` once Joel is merged. |

---

## What I (Fiyin) added on 30 Apr

In `docs/`:
- `report_outline.md` — **master outline + Google Docs formatting guide**. Apply to the Google Doc first.

In `app/`:
- `README.md` — how to run Mohamed's Gradio app locally + smoke-test sentences for screenshots.

In `reports/results/`:
- `q1_2_vocab_overlap.md` — refreshed with full linguistic-distance discussion
- `q5_1_deployment.md` — full Q5.1 write-up
- `q5_2_efficiency.md` — Q5.2 write-up template + table skeleton
- `q4_error_analysis.md` — Q4 template (now includes optional §4.6 LIME panel)

In `scripts/`:
- `benchmark_inference.py` — Q5.2 latency benchmark
- `q4_extract_errors.py` — pulls 10 misclassifications from LoRA model
- `q4_few_shot_eval.py` — 4-shot prompt evaluation
- `lime_explain.py` — LIME interpretability for any of the three model families
- `build_submission_zip.sh` — final ZIP packager (excludes checkpoints/datasets)

In `requirements.txt`:
- Added `lime>=0.2.0.1` for the explainability path.
