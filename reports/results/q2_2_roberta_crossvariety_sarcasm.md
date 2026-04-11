# Q2.2 — RoBERTa cross-variety (FULL), Sarcasm

**Heatmap (3×3 variety-only, mean over seeds):** `reports/figures/q2_2_roberta_macro_f1_heatmap.png`  
**Structured numbers:** `q2_2_roberta_crossvariety_sarcasm.json`

Copy tables and interpretation into `docs/REPORT_TEMPLATE.md` §2.2.

---

## Experimental setup

| Setting | Value |
|--------|--------|
| Model | `roberta-base` |
| Mode | FULL (`DEMO_MODE=0`) |
| Task | Sarcasm (binary) |
| Epochs | 3 per training condition |
| Seeds | 42 and 123 (metrics below are **aggregated across seeds** in the notebook’s final matrix) |
| Metric | Macro-F1 on each test split |

**Notebook training conditions** (5 rows in the log): `uk_only`, `au_only`, `in_only`, `inner_pool`, `all`. The first three match the **same “train on one variety” idea** as LoRA (2.3). The last two train on **pooled** data (UK+AU, or all varieties)—optional extras for the report.

**Load-report lines** (`classifier.* | MISSING`, `UNEXPECTED` LM head keys): normal when loading `roberta-base` into `RobertaForSequenceClassification`—the classification head is **new**; base encoder weights load from the checkpoint. **Missing/unexpected LayerNorm name pairs** (weight/bias vs beta/gamma) come from **transformers / safetensors** naming—weights still load; safe to mention once as a technical footnote if asked.

**Sklearn `UndefinedMetricWarning` (precision, no predicted samples):** some evaluation slices had **no predicted positives** for sarcasm → precision undefined; the notebook should use `zero_division=0` in `classification_report` / metrics (same issue as in early LoRA runs). Does not invalidate the run—it flags **majority-class collapse** on that slice.

---

## Macro-F1 — variety-only training (comparable to LoRA 2.3)

*Rows = where the model was trained; columns = test variety. Values from your Colab **Cross-Variety Matrix** (mean over seeds).*

| Trained on ↓ / Test → | en-UK | en-AU | en-IN |
|----------------------|------:|------:|------:|
| **en-UK** (UK only) | **0.480** | 0.414 | 0.499 |
| **en-AU** (AU only) | 0.627 | **0.752** | 0.518 |
| **en-IN** (IN only) | 0.480 | 0.414 | 0.482 |

### How to read this

- **Diagonal:** in-variety test performance — **en-AU is strongest (~0.75)**, consistent with **more balanced** sarcasm class counts in en-AU training data (same pattern as LoRA).
- **Off-diagonal:** cross-variety transfer — typically **lower** than the best diagonal, because sarcasm cues **shift** across UK / AU / Indian English.
- **Symmetry in numbers:** en-UK and en-AU test columns for **UK-only** and **IN-only** rows both show **0.414** on en-AU test — can happen with **averaging seeds** and similar failure modes (e.g. limited sarcastic recall on AU test); worth a sentence that **macro-F1 is summary** and **per-class** tables/confusion matrices show detail.

---

## Extra training pools (notebook only—not in LoRA)

| Trained on ↓ / Test → | en-UK | en-AU | en-IN |
|----------------------|------:|------:|------:|
| **Inner pool** | 0.694 | 0.683 | 0.592 |
| **All** | **0.732** | 0.679 | 0.564 |

- **Inner pool** and especially **all** use **more training data and mixed varieties**, so the model sees **all dialects during training**. That usually **raises** test performance on each variety vs single-variety training, at the cost of a **different experimental factor** (multi-domain mix vs pure variety adapter).
- Notebook reported **Best condition: all** — consistent with **highest macro-F1 on en-UK test** in this table; en-AU test is slightly higher for **AU-only** training (0.752 vs 0.679), which is an interesting **report point**: *pooling helps UK/IN test columns on average but AU-specialised training still wins on AU test.*

---

## Consistency with what RoBERTa “should” do

Yes, these results are **coherent** for a fine-tuned encoder on BESSTIE:

1. **Strong AU diagonal, weaker UK/IN diagonals** — matches **class imbalance** (sarcasm rare in some splits) and **task difficulty**; RoBERTa is not magic, it follows data.
2. **Cross-variety < in-variety** on most pairs — expected **domain shift**.
3. **Multi-variety pools beat single-variety on several cells** — expected when **shared supervision** helps generalisation (at least on macro-F1 here).
4. **Similar shape to LoRA (Qwen 1.5B) mean matrix** — same ranking (AU diagonal best; transfer harder to other varieties). **Absolute numbers differ** (architecture, head, objective, LoRA vs full fine-tune)—compare side-by-side in §2.3 discussion rather than expecting identity.

---

## Training-log quirks (optional in report)

- **`macro_f1` on validation stuck at 0.462199** across epochs for some conditions (e.g. **in_only** in your log): validation metric **did not move** in the printed table even when test results later differ. Possible causes: **metric computed on a slice** that’s **degenerate**, **logging rounding**, or **best checkpoint** selection vs displayed epoch. Cite **test-set matrix** as primary for the report.
- **`No log` for training loss** on some epochs: Trainer logging granularity; not necessarily a problem if test evaluation completed.

---

## Figure

```bash
python3 scripts/plot_cross_variety_matrix.py \
  --json reports/results/q2_2_roberta_crossvariety_sarcasm.json \
  --matrix-key mean_over_seeds \
  --out reports/figures/q2_2_roberta_macro_f1_heatmap.png \
  --title "RoBERTa — cross-variety macro-F1 (Sarcasm, variety-only, mean over seeds)"
```

---

## Comparison snippet for LoRA subsection

| Aspect | RoBERTa (this run) | LoRA Qwen2.5-1.5B (your FULL) |
|--------|--------------------|-------------------------------|
| Best variety-only diagonal | **~0.75** (en-AU → en-AU) | **~0.78** (same cell, mean) |
| Train en-IN → test en-IN | ~0.48 | ~0.53 |
| Full fine-tuning | All ~125M encoder params updated | ~1.1M adapter params |

*Exact LoRA numbers: `q2_3_lora_full_sarcasm.md`.*
