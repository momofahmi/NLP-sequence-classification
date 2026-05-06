# `main.ipynb` plan — answers to Mohamed's three questions

This is the canonical mapping the team agreed on for the submission notebook. It answers Mohamed's three questions for every section so that anyone (incl. the marker) can run `main.ipynb` end-to-end and reproduce all the numbers in the report.

---

## Mohamed's questions — short answers up front

> **Q1: Is your notebook running everything from scratch to result?**
>
> Each section runs end-to-end. For the heavy training cells (RoBERTa cross-variety, LoRA per-variety) we ship two paths:
> - **Default path (`RETRAIN=False`)** — load the trained model/adapter from the HuggingFace Hub and report the numbers from `reports/results/*.json`. Runs in ~10 minutes on a T4.
> - **Full-retrain path (`RETRAIN=True`)** — re-run the training inline. Total runtime ≈ 60–80 min on a T4.
>
> Fast sections (EDA, vocab, LR + TF-IDF, SVM, error analysis, efficiency) always run from scratch.

> **Q2: Which specific points or sections from your respective domains do you want me to keep in the final notebook?**
>
> See the section-by-section table below.

> **Q3: Since we have several adapter models for the classifications, could you point out exactly which notebook/model you want me to include in the main file?**
>
> **Canonical adapter base = `facebook/opt-1.3b`** with one LoRA per variety, trained by Mohamed.
> Reasoning (already in the report §3.4): OPT-1.3B beats both LLaMA bases on every variety.
> The LLaMA-1B and LLaMA-3.2-3B notebooks are **not** included in `main.ipynb`. They are kept in the repo as `notebooks/lora_notebook_llama1B.ipynb` and `notebooks/lora_notebook_llama3.2B_v2.ipynb` for the appendix table (Table 8 — frozen base comparison) and for marker-curiosity only.

---

## Section-by-section canonical source

| Report section | Canonical source notebook | Owner | Inlined or imported into `main.ipynb` |
|---|---|---|---|
| 0 Smoke test | `notebooks/0_DataLoader_Check_Group.ipynb` | Mohamed | Inlined as 1 cell |
| 1.1 EDA | `notebooks/1.1_EDA_Distributions_Yusrah_Omar.ipynb` | Yusrah + Omar | Inlined (uses `src/eda_distributions.py`) |
| 1.2 Vocab | `notebooks/2.3_LoRA_Preparation_Omar.ipynb` | Omar | Inlined (uses `src/vocabulary_overlap.py`, `linguistic_feature_analysis.py`) |
| 2.1 LR + TF-IDF | `notebooks/2.1_Baseline_TFIDF_LogReg_Yusrah_Omar.ipynb` | Yusrah + Omar | Inlined |
| 2.1 SVM | `notebooks/SVM_TFIDF.ipynb` (origin/main) | Omar | Inlined as 1 cell (LinearSVC sanity check) |
| 2.1 RoBERTa all-pool | `notebooks/2.2_RoBERTa_CrossVariety_Joel_Fiyin.ipynb` (5-condition matrix; we only call the `all` condition for §2.1) | Joel + Fiyin | Inlined; `RETRAIN_ROBERTA` flag |
| 2.2 RoBERTa cross-variety | same notebook, all 5 conditions | Joel + Fiyin | Inlined; `RETRAIN_ROBERTA` flag |
| 2.3 LoRA | `notebooks/2.3_LoRA_Adapters_Mohamed.ipynb` | Mohamed | Inlined; `RETRAIN_LORA` flag; loads adapters from `momofahmi/*` on Hub by default |
| 3.x Evaluation | (no new training) | Fiyin | Inlined — loads `reports/results/*.json` and renders all tables / heatmaps |
| 4 Error analysis | `notebooks/q4_error_analysis_v2.ipynb` + `scripts/q4_extract_errors.py` + `scripts/q4_few_shot_eval.py` | Mohammad | Inlined (calls scripts via `%run`) |
| 5.1 Deployment | `app/app.py` (Gradio) | Mohamed F. | Documented only — pointer + screenshots; do not launch app inside `main.ipynb` |
| 5.2 Efficiency | `scripts/benchmark_inference.py` | Fiyin | Inlined |

---

## Top-of-notebook config flags

```python
# main.ipynb — config
SEEDS = [42, 123]
DEMO_MODE = False               # True = use 200-row subset (smoke run)
RETRAIN_ROBERTA = False         # True = retrain all 5 conditions (~25 min on T4)
RETRAIN_LORA    = False         # True = retrain en-UK / en-AU / en-IN adapters (~30 min on T4)
RETRAIN_LR      = True          # always cheap, leave True
RUN_BENCHMARK   = True          # §5.2 latency table
RUN_LIME        = False         # bonus interpretability — slower
HF_HUB_USER     = "momofahmi"   # adapters live here
DATASET_ID      = "surrey-nlp/BESSTIE-CW-26"
```

When `RETRAIN_*=False`, the notebook downloads the adapters / fine-tuned weights from the Hub and only runs evaluation. This is what the marker should run.

---

## Files included with the submission ZIP

Per the screenshot rules:

- **Inside the ZIP** (mandatory `main.ipynb`):
  - `notebooks/main.ipynb` — entry point
  - `notebooks/0_DataLoader_Check_Group.ipynb` — sanity-check helper
  - `notebooks/1.1_EDA_Distributions_Yusrah_Omar.ipynb`
  - `notebooks/2.1_Baseline_TFIDF_LogReg_Yusrah_Omar.ipynb`
  - `notebooks/2.2_RoBERTa_CrossVariety_Joel_Fiyin.ipynb`
  - `notebooks/2.3_LoRA_Adapters_Mohamed.ipynb`
  - `notebooks/2.3_LoRA_Preparation_Omar.ipynb`
  - `notebooks/q4_error_analysis_v2.ipynb`
  - `notebooks/lora_notebook_llama1B.ipynb` (appendix)
  - `notebooks/lora_notebook_llama3.2B_v2.ipynb` (appendix)
  - `src/` — utility modules
  - `scripts/` — `benchmark_inference.py`, `q4_extract_errors.py`, `q4_few_shot_eval.py`, `lime_explain.py`, `sanitize_notebook.py`
  - `app/` — `app.py`, `streamlit_app.py`, `README.md` (deployment)
  - `requirements.txt`
  - `README.md`
  - `reports/results/*.json` (so evaluation runs in `RETRAIN=False` mode)
  - `reports/figures/*.png` (so the notebook can `display()` them without recomputing)
  - `models/` — Python source only (no checkpoints)
- **Excluded** (size / coursework rule):
  - `runs/`, any `checkpoint-*/`
  - `adapters/*/checkpoint-*/`
  - `notebooks/models/`, `notebooks/tokenized/`
  - `.cache/`
  - `dist/`, build artefacts
  - The PDF report (submitted separately as per the rules)

`scripts/build_submission_zip.sh` already encodes these exclusions.
