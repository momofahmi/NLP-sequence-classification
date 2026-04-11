# BESSTIE: Sentiment & Sarcasm Classification across English Varieties

**Module:** COMM061 Natural Language Processing — Group coursework  
**Dataset:** [`surrey-nlp/BESSTIE-CW-26`](https://huggingface.co/datasets/surrey-nlp/BESSTIE-CW-26)  
**Repository branch (modelling / results):** `fiyin/model-pipeline`

**TODO before submission:** group name, member names, student IDs, declaration of originality (per SurreyLearn), and any sections marked *TODO* below.

---

## Abstract (optional)

We study **sarcasm** (primary) and **sentiment** (baseline notebook) classification across **en-UK**, **en-AU**, and **en-IN** using a **TF‑IDF + logistic regression** baseline, **RoBERTa-base** with a **cross-variety evaluation matrix**, and **LoRA adapters** on a **Qwen2.5-1.5B** backbone. **Macro-F1** is the main metric (class imbalance). Results show **stronger in-variety than cross-variety** performance, **en-AU** often easiest to model for sarcasm under full training, and **parameter-efficient LoRA** achieving **comparable diagonal** macro-F1 to RoBERTa on the same 3×3 protocol while updating **~0.07%** of parameters.

---

## 1. Dataset analysis and visualisation

### 1.1 Label distributions

**Figures (repo):** under `reports/figures/` — e.g. `q1_1_sarcasm_by_variety.png`, `q1_1_sentiment_by_variety.png`, `q1_1_split_distribution.png`, `q1_1_variety_distribution.png`, `q1_1_source_by_variety.png`, train percentages `q1_1_train_sarcasm_by_variety_pct.png`, `q1_1_train_sentiment_by_variety_pct.png`.

**Observations:**

- **Sarcasm** is **minority** in aggregate and varies by variety; models that optimise accuracy alone can **ignore** sarcastic posts — we report **macro-F1** and use **class weights** where noted in training notebooks.
- **Sentiment** is relatively more balanced; baseline **macro-F1** is much higher for sentiment than sarcasm (§2.1).

**Split sizes (dataset):** train 3,747 / validation 313 / test 2,183 rows (full BESSTIE CW-26).

### 1.2 Vocabulary overlap and linguistic distance

**Method:** Jaccard similarity on word sets; TF‑IDF cosine similarity on **concatenated documents per variety** (details in notebook `1.1` / `1.2` and `reports/results/q1_2_vocab_overlap.md`).

**Results (stored run):**

| Measure | en-AU ↔ en-UK | en-AU ↔ en-IN | en-IN ↔ en-UK |
|---------|---------------|---------------|---------------|
| Jaccard | 0.268 | 0.224 | 0.236 |
| TF‑IDF cosine | **0.887** | 0.748 | 0.803 |

**Interpretation:** **Inner-circle** pairs (AU–UK) show **highest** lexical similarity; **en-IN** is more distant in both measures — consistent with **code-mixing**, sources, and spelling/pragmatic differences. This supports expecting **harder cross-variety transfer** when models latch onto variety-specific n-grams.

**Figure:** `reports/figures/q1_2_vocabulary_similarity_heatmap.png`

---

## 2. Experiments

### Global settings

| Item | Value |
|------|--------|
| Primary metric | **Macro-F1** (sarcasm report focus) |
| Seeds (RoBERTa / LoRA FULL) | **42** and **123** |
| RoBERTa cross-variety table | Mean over seeds (notebook-aggregated matrix) |
| LoRA | Per-seed tables + **mean** matrix in `reports/results/q2_3_lora_full_sarcasm.md` |
| Imbalance | Class-weighted loss (LoRA); see RoBERTa notebook for its objective |

### 2.1 Baseline — TF‑IDF + Logistic Regression

**Notebooks:** `notebooks/2.1_Baseline_TFIDF_LogReg_Yusrah_Omar.ipynb`  
**Details:** TF‑IDF features + logistic regression; see notebook for `max_features`, n-grams.

**Test macro-F1 (primary comparison row):**

| Task | Accuracy | Macro-F1 |
|------|----------|----------|
| **Sarcasm** | 0.770 | **0.407** |
| Sentiment | 0.819 | **0.812** |

Validation metrics are tabulated in `reports/results/q2_1_baseline_tfidf.md`.

**Analysis:** Bag-of-words features capture **lexical** overlap but miss much **pragmatic** sarcasm — sarcasm **macro-F1** stays well below sentiment. This sets a **strong, simple baseline** for transformer and LoRA sections.

### 2.2 RoBERTa — cross-variety evaluation (Sarcasm)

**Notebook:** `notebooks/2.2_RoBERTa_CrossVariety_Joel_Fiyin.ipynb`  
**Model:** `roberta-base`, FULL mode, 3 epochs, GPU.

**Variety-only training (3×3, comparable to LoRA)** — macro-F1 on test, **mean over seeds**:

| Trained on ↓ / Test → | en-UK | en-AU | en-IN |
|----------------------|------:|------:|------:|
| en-UK | 0.480 | 0.414 | 0.499 |
| en-AU | 0.627 | **0.752** | 0.518 |
| en-IN | 0.480 | 0.414 | 0.482 |

**Additional pools from the same notebook** (mixed-variety training — *not* part of the LoRA adapter protocol):

| Trained on ↓ / Test → | en-UK | en-AU | en-IN |
|----------------------|------:|------:|------:|
| Inner pool | 0.694 | 0.683 | 0.592 |
| All | **0.732** | 0.679 | 0.564 |

The notebook reported **best overall condition: all** (highest macro-F1 on several tests); **AU-only** still wins **en-AU test** vs **all** (0.752 vs 0.679).

**Figures:**  
- `reports/figures/q2_2_roberta_macro_f1_heatmap.png` (variety-only 3×3)  
- `reports/figures/q2_2 (RoBERTa) Cross-Variety Evaluation Matrix - Macro-F1.png` (if used in slides)  
- `reports/figures/q2_2 (RoBERTa) Confusion Matrix.png`

### 2.3 LoRA — Qwen2.5-1.5B adapters (Sarcasm)

**Notebook:** `notebooks/2.3_LoRA_Adapters_Mohamed.ipynb`  
**Base model:** `Qwen/Qwen2.5-1.5B`  
**LoRA (defaults in `models/lora/lora_adapters.py`):** rank **r = 8**, **α = 16**, dropout **0.1**, target modules **`q_proj`, `v_proj`**, **~1.09M trainable** parameters (~**0.07%** of ~1.54B total).  
**Training:** FULL (`DEMO_MODE=0`), **3 epochs**, seeds **42** & **123**, weighted cross-entropy.

**Mean macro-F1 over seeds (test):**

| Adapter trained on ↓ / Test → | en-UK | en-AU | en-IN |
|-------------------------------|------:|------:|------:|
| en-UK | 0.564 | 0.460 | 0.560 |
| en-AU | 0.614 | **0.775** | 0.528 |
| en-IN | 0.498 | 0.416 | 0.529 |

Per-seed tables and interpretation (seed instability on en-UK adapter, en-IN difficulty) are in `reports/results/q2_3_lora_full_sarcasm.md`.

**Figure:** `reports/figures/q2_3_lora_macro_f1_heatmap.png`

### 2.4 RoBERTa vs LoRA (same 3×3 protocol)

| Aspect | RoBERTa variety-only (mean) | LoRA (mean) |
|--------|------------------------------|-------------|
| Best diagonal (en-AU → en-AU) | **0.752** | **0.775** |
| Train en-IN → test en-IN | 0.482 | 0.529 |
| Parameters updated | Full encoder fine-tune (~125M) | LoRA only (~1.1M) |

Differences also reflect **model family** (encoder vs causal LM + classification head) and **optimisation noise**; both show the **same qualitative pattern**: **AU diagonal strongest**, **cross-variety** harder than **in-variety**.

---

## 3. Summary evaluation

| Model | Role | Sarcasm test macro-F1 (typical headline) |
|-------|------|------------------------------------------|
| TF‑IDF + LogReg | Baseline | **0.407** (overall test; not variety-split) |
| RoBERTa | Cross-variety matrix | **up to ~0.75** in-variety (AU); see §2.2 |
| LoRA Qwen1.5B | Cross-variety matrix | **up to ~0.78** in-variety (AU mean); see §2.3 |

**Failure modes:** **Majority-class bias**, **domain shift** between varieties, and **minority sarcasm** recall — see confusion matrices in notebooks and sklearn warnings where **no** sarcastic predictions occurred on a slice.

---

## 4. Error analysis and few-shot prompting

**TODO (group):** Select **10** errors from the best large model / adapter; linguistically analyse **4**; build a **4-shot prompt**; re-test **6** remaining errors; report before/after. *(No fixed numbers in-repo yet.)*

---

## 5. Deployment and efficiency

**TODO (group):** Streamlit / Gradio app (`app/` or agreed path) — variety selector, model or adapter routing, **screenshots**.  
**TODO:** Rough **latency** comparison (baseline vs RoBERTa vs LoRA) — small vs longer inputs.

---

## References

- BESSTIE dataset: `surrey-nlp/BESSTIE-CW-26` (Hugging Face).  
- RoBERTa: Liu et al., RoBERTa (2019).  
- LoRA: Hu et al., LoRA (2021).  
- Libraries: Hugging Face `transformers`, `datasets`, `peft`, `scikit-learn`, `PyTorch` (see `requirements.txt`).

---

## Appendix: Source-of-truth files in the repo

| Content | Location |
|---------|----------|
| Baseline metrics | `reports/results/q2_1_baseline_tfidf.md`, `reports/results/local_run_summary.json` |
| Vocab overlap | `reports/results/q1_2_vocab_overlap.md` |
| RoBERTa matrices | `reports/results/q2_2_roberta_crossvariety_sarcasm.md`, `.json` |
| LoRA matrices | `reports/results/q2_3_lora_full_sarcasm.md`, `.json` |
| Heatmap scripts | `scripts/plot_cross_variety_matrix.py` |
| Index | `reports/results/README.md` |

*End of draft. Export to PDF per module formatting rules.*
