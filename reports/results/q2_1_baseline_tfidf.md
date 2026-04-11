# Q2.1 — Baseline TF‑IDF + Logistic Regression

**Source:** metrics consolidated in `local_run_summary.json` (from notebook `2.1_Baseline_TFIDF_LogReg_Yusrah_Omar.ipynb`). Update the JSON if you re-run and numbers change.

## Setup (for the report)

- **Features:** TF‑IDF on text (see notebook for `max_features`, n‑grams, etc.).
- **Classifier:** logistic regression.
- **Splits:** BESSTIE train / val / test as in the notebook.

## Validation metrics

| Task | Accuracy | Macro precision | Macro recall | Macro F1 |
|------|----------|-----------------|--------------|----------|
| Sarcasm | 0.783 | 0.338 | 0.568 | **0.424** |
| Sentiment | 0.843 | 0.833 | 0.850 | **0.841** |

## Test metrics

| Task | Accuracy | Macro precision | Macro recall | Macro F1 |
|------|----------|-----------------|--------------|----------|
| Sarcasm | 0.770 | 0.318 | 0.564 | **0.407** |
| Sentiment | 0.819 | 0.824 | 0.801 | **0.812** |

## Interpretation (short)

- **Sentiment** is easier than **sarcasm** under a bag‑of‑words model: sarcasm depends heavily on **context, irony, and pragmatics**, which sparse lexical features capture poorly — hence lower macro-F1 and precision on sarcasm despite reasonable recall.
- Use this table as the **lower baseline** when comparing to RoBERTa (2.2) and LoRA (2.3) on the **same task** (apples‑to‑apples: sarcasm vs sarcasm).
