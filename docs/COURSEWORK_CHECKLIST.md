# Coursework checklist

Tick items by editing this file: `[ ]` → `[x]`. Align with `docs/report.md` (draft) and `docs/REPORT_TEMPLATE.md` (section map).

---

## Data & EDA (Q1)

- [x] BESSTIE loaded; train/val/test splits understood
- [x] Q1.1 EDA figures under `reports/figures/` (`q1_1_*`)
- [x] Q1.2 vocabulary / overlap + figure (`q1_2_vocabulary_similarity_heatmap.png`; write-up: `reports/results/q1_2_vocab_overlap.md`)

---

## Baseline (Q2.1)

- [x] TF-IDF + LogReg run with metrics (`notebooks/2.1_...`, `reports/results/q2_1_baseline_tfidf.md`)
- [x] Key metrics captured for report

---

## RoBERTa cross-variety (Q2.2)

- [x] Notebook `2.2` FULL (`DEMO_MODE=0`), GPU
- [x] Seeds 42 & 123; **mean** cross-variety matrix documented (`reports/results/q2_2_roberta_crossvariety_sarcasm.md` + `.json`)
- [x] Figures: `reports/figures/q2_2_roberta_macro_f1_heatmap.png`; optional originals `q2_2 (RoBERTa) *.png`
- [ ] **PDF only:** paste §2.2 prose + tables from `docs/report.md` into final document; add **group** discussion of transfer vs `inner_pool` / `all` if required by brief

---

## LoRA (Q2.3)

- [x] Notebook `2.3`; base **Qwen2.5-1.5B**; GPU
- [x] FULL run (`DEMO_MODE=0`); seeds 42 & 123; 3 variety adapters + eval grid
- [x] Results archived: `reports/results/q2_3_lora_full_sarcasm.md` + `.json`
- [x] Heatmap: `reports/figures/q2_3_lora_macro_f1_heatmap.png`
- [ ] **PDF only:** include **1–2 confusion matrices** (best diagonal + hard transfer) from notebook output if brief asks for per-class detail

---

## Report & submission

- [x] Draft report content compiled: **`docs/report.md`** (merge with Yusrah’s doc / house style)
- [ ] Final **PDF** per module rules (page limits per `REPORT_TEMPLATE.md` sections)
- [ ] Title page: **group name**, **all members**, **declaration of originality**
- [ ] **References** complete; figures cited in text
- [ ] Code **ZIP** / bundle per brief (exclude large `tmp/` checkpoints, `.venv`, etc.)
- [ ] **Group review:** contributions paragraph + proofread

---

## Error analysis & few-shot (if required)

- [ ] §4 in `REPORT_TEMPLATE.md`: 10 errors, 4 linguistic analyses, 4-shot prompt, re-test 6 — *TODO in `docs/report.md`*

---

## Deployment & efficiency (if required)

- [ ] Demo app runs (`app/streamlit_app.py` or agreed) + **screenshots**
- [ ] Latency / efficiency paragraph (baseline vs RoBERTa vs LoRA) — *TODO in `docs/report.md`*

---

## Optional / hygiene

- [ ] `README.md` Colab links match branch you submit
- [ ] Private repo: collaborators + Colab `GITHUB_TOKEN` documented for reproducibility

---

*Last updated: draft report `docs/report.md`; modelling branch `fiyin/model-pipeline`.*
