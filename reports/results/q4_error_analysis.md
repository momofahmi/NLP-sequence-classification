# Q4 — Error Analysis & Few-Shot Prompting (10 marks, max 4 pages)

> Report-ready template. The two scripts below produce all the data needed to fill the placeholders. **Mohammad is the assigned author** (per the docx). Once filled, paste this into the docx under §4.

---

## How to populate this section

```bash
# 1. Pull 10 misclassifications from the best LoRA model (mix of FPs/FNs across varieties).
python scripts/q4_extract_errors.py \
    --base-llm  facebook/opt-1.3b \
    --task      Sarcasm \
    --n-errors  10 \
    --out       reports/results/q4_errors.json

# 2. Open reports/results/q4_errors.json and write a one-paragraph
#    `explanation` for any 4 of the 10 examples. Save it.

# 3. Test the remaining 6 with the few-shot prompt built from those 4.
python scripts/q4_few_shot_eval.py \
    --in           reports/results/q4_errors.json \
    --judge-model  Qwen/Qwen2.5-1.5B-Instruct \
    --task         Sarcasm \
    --out          reports/results/q4_fewshot_results.json
```

The second script prints the before/after comparison and saves a JSON. Drop the numbers into Tables 4.1 and 4.2 below.

---

## §4 — text to paste into the docx

We extracted ten test-set examples that the best LoRA adapter (OPT-1.3B + variety-specific LoRA) got wrong, balanced across the three varieties and across both error types (false positives and false negatives). The full list with metadata is in `reports/results/q4_errors.json`.

### 4.1 The ten misclassifications

`[FILL with a short table — see Table 4.1.1 below. The script writes the JSON; copy the relevant fields.]`

### Table 4.1.1 — Ten misclassifications from the LoRA model on the Sarcasm task

| # | Variety | Source | Gold | Pred | Text |
|---|---|---|---|---|---|
| 1 | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| 2 | … | | | | |
| 10 | … | | | | |

### 4.2 Four examples — linguistic explanation

For four of these examples we provide a written explanation of why the text is (or is not) sarcastic. These four also serve as the in-context exemplars for the few-shot prompt in §4.3. We picked them to span the three categories of features that BESSTIE's own error analysis (Srirag et al. 2025, Table 8) flags as systematically hard:

- **DIAL** — pervasive dialect features (article omission, copula drop in en-IN; pronoun drop in all three).
- **COLL** — locale-specific colloquialisms (en-AU *“arvo”*, *“Hospo workers”*; en-UK *“warra”*).
- **CONT** — contextual / world-knowledge cues (named entities, references to ongoing events).
- **CODE** — code-mixed Hindi tokens (en-IN only).

#### Example A — `[variety]`, gold = `[Sarcastic / Not Sarcastic]`, model predicted = `[opposite]`

> *“`[insert text from q4_errors.json]`”*

**Why the model was wrong:** `[fill, ~3–5 sentences. Discuss what specific dialect/colloquial/code-mix/context feature the model missed. E.g. for an en-AU example: the irony hinges on the colloquial use of "good onya, mate" as a positive phrase ironically applied to bad behaviour — a feature category COLL × CONT — which the model treats as straightforwardly positive because the words themselves are positive.]`

#### Example B — `[…]`

> *“`[…]`”*

**Why the model was wrong:** `[fill]`

#### Example C — `[…]`

> *“`[…]`”*

**Why the model was wrong:** `[fill]`

#### Example D — `[…]`

> *“`[…]`”*

**Why the model was wrong:** `[fill]`

### 4.3 The four-shot prompt

The four explanations above are concatenated with their gold labels into a single instruction-tuned prompt (template in `scripts/q4_few_shot_eval.py → build_prompt()`). The prompt format is:

```
You are a text classifier. For each input, output exactly one label
(Sarcastic or Not Sarcastic) followed by a one-sentence explanation.
Use the cultural and linguistic context of the indicated English variety.

Variety: <variety>
Text: <text>
Label: <gold label>
Explanation: <our written explanation>

(... × 4 exemplars ...)

Variety: <target variety>
Text: <target text>
Label:
```

### 4.4 Re-testing the remaining six examples

We submitted the remaining six misclassifications to a separate generative judge model (`[Qwen/Qwen2.5-1.5B-Instruct or whichever model you actually used]`) using the four-shot prompt above, and parsed the generated label.

### Table 4.4.1 — LoRA (no prompt) vs four-shot judge model on the same six examples

| # | Variety | Gold | LoRA pred | 4-shot pred | Improved? |
|---|---|---|---|---|---|
| 5 | `[fill]` | `[…]` | `[…]` | `[…]` | `[Y/N]` |
| 6 | … | | | | |
| 10 | … | | | | |

**Aggregate:** before few-shot, the LoRA model was correct on `[0]/6` of the held-out misclassifications (by definition — these were chosen as errors). After the four-shot prompt, the judge model was correct on **`[X]/6`**.

### 4.5 Discussion

`[fill, ~6–10 lines. Suggested talking points:]`

- **Did the prompt help?** State the delta. If `Δ ≥ 2`, frame this as the few-shot prompt successfully transferring the variety-specific cues we wrote into a stronger model. If `Δ ≈ 0`, frame this honestly: the failures concentrate on `[which feature category — code-mixing? cultural reference?]`, which a single short prompt cannot teach.
- **Which examples flipped to correct?** Most likely the ones whose error category matches one of the four exemplars (e.g. if exemplar B explains code-mixing, the en-IN code-mixed test example is the most likely to flip).
- **Which examples still fail?** Almost certainly any example whose hardness comes from world-knowledge / a reference to a current event the judge model has no context for. This is the boundary that no amount of in-context learning can cross without retrieval.
- **Compared to a naïve zero-shot run** (judge model with no exemplars at all): we expect the four-shot to outperform zero-shot, because the exemplars commit the model to producing exactly one of the two labels in a parseable format and ground it in the variety.
- **What this says about the LoRA model.** Where the few-shot prompt succeeds and the LoRA model fails, it is evidence that the *base* LoRA's parametric memory of variety-specific cues is shallower than what a generative model can recover from a handful of in-context examples — a strong argument for hybrid systems (parametric + retrieval/prompting) at deployment time.

---

### 4.6 Optional — LIME interpretability for two of the four exemplars

The brief explicitly allows LIME / SHAP for "interpretable evaluation of the errors made by models" in §2.2. We use **LIME** because it is model-agnostic and works equally well across the three model families (TF-IDF + LR, RoBERTa, OPT-1.3B + LoRA) using the same `predict_proba` interface.

```bash
# Generate token-importance plots for the 4 explained examples, using the
# variety-specific LoRA adapter for each one.
python scripts/lime_explain.py \
    --model    lora \
    --in       reports/results/q4_errors.json \
    --out-dir  reports/figures/lime/

# Compare what the classical baseline keys on, on the same sentences.
python scripts/lime_explain.py \
    --model      tfidf \
    --in         reports/results/q4_errors.json \
    --out-dir    reports/figures/lime/
```

This produces `reports/figures/lime/lime_lora_<variety>_<i>.png` and `..._tfidf_<variety>_<i>.png` for each of the four explained examples, plus a `lime_<model>_summary.json` with the top-10 token weights per example.

`[FIGURE 4.6.1 — side-by-side LIME panels for one en-AU example: top panel = OPT-1.3B+LoRA(en-AU), bottom = TF-IDF+LR. Compare which tokens drive the prediction.]`

**Suggested discussion (1 paragraph in the docx):** the LoRA model attends to whole-phrase patterns (e.g. *"good onya, mate"* lights up together as a positive-surface ironic-content pair), whereas TF-IDF+LR's attribution is purely lexical (*"legend"* and *"mate"* push individual probabilities up or down with no awareness of how they combine). This is concrete evidence for the architectural / pre-training argument made in §2.1: the contextual model captures the *incongruity* that defines sarcasm, the bag-of-words model cannot.

---

## Reproducibility note

All of the inputs are pulled programmatically from `surrey-nlp/BESSTIE-CW-26` (test split). The misclassification set in `q4_errors.json` is deterministic given `--seed 42`, and the few-shot evaluation is run with `do_sample=False`, so the exact numbers in the tables above can be regenerated by re-running the two scripts.
