# Branch `fiyin/model-pipeline` — summary for the group

This document describes what was integrated on this branch, how the coursework pipeline maps to the repo, and what to run next (especially on **Google Colab** for RoBERTa and LoRA).

---

## 1. Purpose

Bring the BESSTIE coursework work into one coherent structure:

- **Q1** — Dataset EDA and vocabulary overlap (figures + metrics).
- **Q2.1** — TF‑IDF + Logistic Regression baseline vs (later) RoBERTa.
- **Q2.2** — RoBERTa cross-variety evaluation matrix (full runs on **GPU**).
- **Q2.3** — LoRA adapters per variety (full runs on **GPU**).
- **Q5** — Streamlit demo app (placeholder models until fine-tuned weights are wired in).

---

## 2. Branch name

- **Branch:** `fiyin/model-pipeline`
- **Base:** evolved from `main` with notebook renames, `src`/`models` layout, Colab-friendly cells, and local smoke tests.

---

## 3. Repository layout (high level)

| Area | Contents |
|------|----------|
| `notebooks/` | Numbered notebooks by task + owner names (see below). |
| `src/` | Data loading, EDA, TF‑IDF features, vocab overlap, training helpers, tokenization utils. |
| `models/baseline/` | `logistic_regression.py` + saved baseline weights (see `.gitignore` if excluded). |
| `models/lora/` | `lora_adapters.py` — LoRA config, load/train helpers, supported base model keys. |
| `models/tfidf/` | TF‑IDF vectorizer pickle; sparse matrices may be generated locally (often gitignored). |
| `scripts/` | `smoke_roberta.py`, `smoke_lora.py` — quick local pipeline checks. |
| `app/streamlit_app.py` | Simple UI for variety + task (plug in real model IDs later). |
| `reports/figures/` | Q1.1 / Q1.2 figures (`q1_1_*`, `q1_2_*` naming). |
| `reports/results/` | e.g. `local_run_summary.json` with baseline + vocab numbers from local runs. |
| `docs/` | This file + `REPORT_TEMPLATE.md` for the PDF report. |

---

## 4. Notebooks (task ↔ file)

| Notebook | Role |
|----------|------|
| `0_DataLoader_Check_Group.ipynb` | Sanity check BESSTIE loads. |
| `1.1_EDA_Distributions_Yusrah_Omar.ipynb` | Q1.1 distributions (EDA class). |
| `2.1_Baseline_TFIDF_LogReg_Yusrah_Omar.ipynb` | Q2.1 baseline. |
| `2.2_RoBERTa_CrossVariety_Joel_Fiyin.ipynb` | RoBERTa + cross-variety matrix; **first cell = Colab setup**. |
| `2.3_LoRA_Preparation_Omar.ipynb` | Prep / vocab / linguistic exploration. |
| `2.3_LoRA_Adapters_Mohamed.ipynb` | LoRA training loop; **first cell = Colab setup**. |

**Colab:** open from GitHub via *Open in Colab* (URLs in `README.md` use `momofahmi/NLP-sequence-classification`; change if the repo moves).

---

## 5. Key `src` modules (renamed for clarity)

| Module | Role |
|--------|------|
| `besstie_data_loader.py` | `load_besstie`, `get_BESSTIE_splits`, `get_variety_split`, `get_train_conditions` / `get_test_conditions` for cross-variety experiments. |
| `eda_distributions.py` | Plots; saves under `reports/figures/`. |
| `vocabulary_overlap.py` | Jaccard + TF‑IDF cosine + heatmap. |
| `tfidf_feature_extraction.py` | Fit/transform TF‑IDF; writes under `models/tfidf/`. |
| `training_utils.py` | Class-weighted `Trainer` helper for LoRA. |
| `transformer_tokenization_utils.py` | Tokenization helpers for prep notebooks. |
| `linguistic_feature_analysis.py` | Linguistic plots (when spaCy model available). |

---

## 6. What was run **locally** (already done)

- **Q1.1:** EDA figures with consistent names, e.g. `reports/figures/q1_1_*.png`.
- **Q1.2:** Vocabulary similarity heatmap `reports/figures/q1_2_vocabulary_similarity_heatmap.png` + Jaccard / cosine tables (also summarized in `reports/results/local_run_summary.json`).
- **Q2.1:** Full train/val/test TF‑IDF + multi-output LogReg; metrics in `local_run_summary.json`.
- **Smoke tests:** `scripts/smoke_roberta.py` and `scripts/smoke_lora.py` — verify training/eval **runs**; they use small data / short training and are **not** the final coursework numbers.

---

## 7. What must run on **Colab (GPU)**

Per brief: two seeds, proper epochs, macro‑F1 + per-class metrics, confusion matrices.

1. **`2.2_RoBERTa_CrossVariety_Joel_Fiyin.ipynb`**
   - Runtime: **GPU**.
   - Run Colab setup cell (clone repo, `pip install -r requirements.txt`).
   - Run full experiment loop (sarcasm or sentiment — brief allows one task for the matrix; sarcasm recommended).
   - Save `results/*.json`, `figures/*.png` from the notebook into the repo or download for the report.

2. **`2.3_LoRA_Adapters_Mohamed.ipynb`**
   - Runtime: **GPU**.
   - Use a **1B–3B** open base (e.g. `MODEL_KEY=qwen2.5-1.5b` in the notebook / env).
   - Train **three** adapters (en‑UK, en‑AU, en‑IN), evaluate on each test set, log tables + confusion matrices.

Optional: `HF_TOKEN` in Colab secrets for higher Hub rate limits and gated models.

---

## 8. Submission reminders (coursework)

- Do **not** put the PDF inside the code ZIP.
- Avoid zipping **trained checkpoints** / huge binaries; regenerate from notebooks if needed.
- Notebook(s) should be the main evidence; report PDF references figures and tables.

---

## 9. Quick commands (local)

```bash
# Baseline smoke / full baseline: use notebook 2.1 or prior Python one-liners in history
python3 scripts/smoke_roberta.py
python3 scripts/smoke_lora.py

# Streamlit demo
streamlit run app/streamlit_app.py
```

---

## 10. Contact / ownership

This branch was prepared for integration and presentation by **Fiyin** with contributions merged from Joel’s RoBERTa notebook path, Yusrah/Omar baseline and EDA, and Mohamed’s LoRA direction. Adjust names in the report to match your group’s declaration.

---

*Last updated: aligned with branch `fiyin/model-pipeline` before Colab full runs for RoBERTa + LoRA.*
