# Coursework checklist

Tick items in GitHub by editing this file: change `[ ]` to `[x]`. You can also copy the list into Issues / Projects if the group prefers.

## Data & EDA (Q1)

- [ ] BESSTIE loaded; train/val/test splits understood
- [ ] Q1.1 EDA figures produced and stored under `reports/figures/` (e.g. `q1_1_*`)
- [ ] Q1.2 vocabulary / overlap analysis + figure (e.g. `q1_2_*`)

## Baseline (Q2.1)

- [ ] TF-IDF + LogReg (or agreed baseline) run with reported metrics
- [ ] Key metrics captured for report (e.g. in `reports/results/` or notebook output)

## RoBERTa cross-variety (Q2.2)

- [ ] Notebook `2.2` run in **FULL** mode (`DEMO_MODE=0`), GPU
- [ ] Two seeds; macro-F1 cross-variety matrix + confusion matrices / JSON results
- [ ] Final figures saved under `reports/figures/` (e.g. `q2_2_*`) for the PDF

## LoRA (Q2.3)

- [ ] Notebook `2.3` run: 1–3B base (e.g. `qwen2.5-1.5b`), GPU
- [ ] **FULL** mode (`DEMO_MODE=0`) for final numbers; DEMO optional for smoke tests
- [ ] Three variety adapters + evaluation grid; results noted for report
- [ ] Figures / tables archived for report if required by brief

## Report & submission

- [ ] PDF report drafted per module rules (figures + tables + discussion)
- [ ] Code ZIP / submission bundle matches brief (exclude huge checkpoints if required)
- [ ] Group review of contributions and references

## Deployment (if required)

- [ ] Demo app (`app/streamlit_app.py` or equivalent) runs locally / hosted
- [ ] Latency or UX notes for report if required

---

*Last updated: align with branch `fiyin/model-pipeline`.*
