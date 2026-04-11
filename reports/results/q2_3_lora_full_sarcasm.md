# Q2.3 — LoRA adapters (FULL run), Sarcasm

**Heatmap (repo):** `reports/figures/q2_3_lora_macro_f1_heatmap.png` (mean macro-F1 over seeds).

**Generated for the group report.** Copy tables and paragraphs into `docs/report_template.md` §2.3 and §3 as needed.

## Experimental setup

| Setting | Value |
|--------|--------|
| Base model | `Qwen/Qwen2.5-1.5B` (`MODEL_KEY=qwen2.5-1.5b`) |
| Mode | **FULL** (`DEMO_MODE=0`) |
| Epochs | 3 |
| Seeds | 42, 123 |
| Trainable (LoRA) | ~1.09M params (~0.07% of ~1.54B total) |
| Task | Sarcasm (binary) |
| Protocol | Train **one adapter per variety** on that variety’s train split; evaluate each adapter on **en-UK / en-AU / en-IN test** splits (same protocol as RoBERTa 2.2). |
| Loss | Weighted cross-entropy (class weights from training split). |

The notebook’s `score.weight | MISSING` line is **expected**: the pretrained LM has no sequence-classification head; that layer is **randomly initialised** and learned during fine-tuning.

---

## How to read the results dictionary

Structure: `results[train_variety][seed_<id>][test_variety]` → `macro_f1`, `confusion_matrix`, `classification_report`.

- **Rows (adapter):** where the adapter was trained — `en-UK`, `en-AU`, `en-IN`.
- **Columns (test):** which variety’s **test** set was used.
- **Diagonal** (e.g. en-AU adapter → en-AU test): **in-variety** performance.
- **Off-diagonal**: **cross-variety transfer** (domain shift).

**Primary metric (coursework):** **macro-F1** (balances both classes; important when sarcasm is the minority class).

---

## Macro-F1 (test sets)

### Seed 42

| Adapter trained on ↓ / Test → | en-UK | en-AU | en-IN |
|-------------------------------|------:|------:|------:|
| **en-UK** | **0.649** | 0.505 | 0.605 |
| **en-AU** | 0.612 | **0.781** | 0.533 |
| **en-IN** | 0.498 | 0.419 | 0.528 |

### Seed 123

| Adapter trained on ↓ / Test → | en-UK | en-AU | en-IN |
|-------------------------------|------:|------:|------:|
| **en-UK** | 0.480 | 0.414 | 0.516 |
| **en-AU** | 0.616 | **0.769** | 0.522 |
| **en-IN** | 0.498 | 0.413 | 0.529 |

### Mean over seeds (for a single summary matrix)

| Adapter trained on ↓ / Test → | en-UK | en-AU | en-IN |
|-------------------------------|------:|------:|------:|
| **en-UK** | 0.564 | 0.460 | 0.560 |
| **en-AU** | **0.614** | **0.775** | 0.528 |
| **en-IN** | 0.498 | 0.416 | 0.529 |

---

## What these numbers mean (interpretation)

1. **Best in-variety performance is en-AU → en-AU** (macro-F1 ≈ **0.78** averaged over seeds). The en-AU training split is **more class-balanced** (your logs: ~808 vs ~337 sarcastic examples) than en-UK (~1111 vs ~92) or en-IN (~1304 vs ~95), so the model sees **more minority-class signal** and learns a less trivial decision boundary. Confusion matrices for en-AU show **meaningful counts on both classes** (e.g. sarcastic precision/recall in a usable range).

2. **Cross-variety transfer drops off the diagonal**, especially for **sarcastic recall** when testing on another variety. That matches **domain shift** in how sarcasm is expressed across UK / AU / Indian English (lexicon, pragmatics, sources). Macro-F1 on off-diagonal cells is often **0.41–0.62** here — not random, but clearly below the best diagonal cell.

3. **en-IN training is the hardest regime** (heaviest imbalance in the logs: ~1304 vs ~95). Adapters trained on en-IN show **low sarcastic recall** on all test sets (many predictions stuck near “not sarcastic”). One training run (en-IN, seed 42) logged an **extremely low training loss on epoch 3** (~0.0008) while validation loss remained high — a sign of **instability / possible overfitting** to the training distribution; treat that seed’s test metrics with caution and prefer **averaging over seeds** or **early stopping** in future work.

4. **Seed sensitivity:** The **en-UK adapter** with seed **123** **collapsed** to predicting only the majority class on en-UK and en-AU test (confusion matrices with **no** predicted sarcastic examples → sarcastic F1 = 0). Seed **42** did **not** show that collapse for the same adapter. This shows **variance** between initialisations and optimisation paths under imbalance — exactly why the coursework asks for **multiple seeds**.

5. **Identical metrics for two cells:** en-IN adapter, en-UK test, seeds 42 and 123 report **identical** macro-F1 and nearly identical confusion matrices. That usually means both runs **converged to the same effective decision rule** on that test set (e.g. almost always “not sarcastic”), not that the pipeline duplicated results incorrectly.

---

## Training logs (brief)

- **Steps per epoch** differ by variety (train size differs): e.g. ~903 steps (en-UK), ~861 (en-AU), ~1050 (en-IN) with batch size 4 — consistent with **full per-variety training** subsets.
- **Validation loss** generally **decreases** over epochs for en-UK and en-AU; en-IN shows **noisy** behaviour (epoch 2 worse than epoch 1 on seed 42), consistent with **hard imbalance + limited positive examples**.

---

## How to compare with RoBERTa (2.2)

Place the **mean** LoRA matrix above next to the **RoBERTa cross-variety macro-F1 matrix** from notebook 2.2 / `reports/figures/`. Expected discussion points:

- **RoBERTa** fine-tunes **all** encoder weights on the task; **LoRA** updates only a **tiny** adapter. You may see **higher** macro-F1 for RoBERTa on the same grid, or **similar diagonal** but different off-diagonal transfer — either outcome is interpretable if tied to **data per variety**, **capacity**, and **training stability**.

- **Fairness:** Same splits, same metric (macro-F1), same 3×3 protocol — differences are due to **model family** (encoder RoBERTa vs causal Qwen + head) and **fine-tuning method** (full vs LoRA).

---

## Report figures (repo)

1. **Heatmap (generated):** `reports/figures/q2_3_lora_macro_f1_heatmap.png` — produced with:
   ```bash
   python3 scripts/plot_cross_variety_matrix.py \
     --json reports/results/q2_3_lora_full_sarcasm.json \
     --matrix-key mean_over_seeds \
     --out reports/figures/q2_3_lora_macro_f1_heatmap.png \
     --title "LoRA — cross-variety macro-F1 (Sarcasm, mean over seeds)"
   ```
2. **Confusion matrices (from notebook output):** paste or screenshot the **best** diagonal (en-AU → en-AU) and one **hard transfer** pair (e.g. en-IN adapter → en-AU test) into the PDF to show per-class behaviour, not only macro-F1.

---

*Raw structured numbers: `q2_3_lora_full_sarcasm.json` in this folder.*
