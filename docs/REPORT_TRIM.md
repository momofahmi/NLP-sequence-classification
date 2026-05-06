# Report trim guide — get from 29 → 25 pages

> Current state (PG15): 29 pages. Target: 25. Need to lose **4 net**.
> Distribution: §1=5 (−1), §2=10 (−4), §3=8 (−3), §4=3 (ok), §5=3 (ok).
> §4 and §5 have ~3 pages of slack between them, so the **practical target is −7 from §2+§3** combined; you can afford to grow §4 slightly if you want.

This file gives, for every bloated subsection, three things:

1. **What to KEEP** — the load-bearing sentences/figures/tables.
2. **What to CUT** — redundant background, repeated statistics, decorative figures.
3. **What to REPLACE WITH** — a tighter rewrite, ready to paste into the docx.

The biggest wins: (a) consolidating the four separate "class imbalance" paragraphs spread across §1–§3 into one canonical paragraph in §1.1 with backreferences, (b) collapsing the SVM into a 2-sentence aside in §2.1 instead of a full subsection in §2.1 + §3.2, (c) cutting the "Sarcastic Class Gap" sub-discussion of §2.1 (~½ page of repetition), (d) deleting §3.4 subsections that duplicate §2.3.

---

## Section 1 — lose 1 page

### 1.1 Distribution and class imbalance

**Keep:** Figure 1 (variety distribution), Figure 5+6 merged into a single 2-panel figure (sarcasm + sentiment by variety), Figure 8 (sarcasm/sentiment correlation), Figure 12 (POS), and the 97.52% finding.

**Cut:**
- Figures 2, 3, 4 (data sources, source-by-variety, splits) — collapse into **one** small "split + source breakdown" figure. The exact numbers can stay in a 1-line caption.
- Figures 9–11 (one slang figure per variety) — collapse into **one** single panel showing distinctive slang for all three varieties side by side, OR drop entirely and keep the slang discussion in §1.2 only.
- The sentence "Models can become biased due to this imbalance, which is why we then use Macro-F1 as the best metric for evaluation to prevent majority class predictions" — already covered in §3 intro; remove.

**Estimated saving:** ~0.5 pages from figure consolidation, ~0.4 pages from removing the sentence and slang panels = **~0.9 pages**. With the Section 1.2 trim below, total §1 saving is ~1 page.

### 1.2 Vocabulary analysis

**Keep:** Numerical results (Jaccard / TF-IDF cosine), the linguistic-distance definition, the "superficial vs grammatical" verdict, **one** heatmap figure (current Fig 13).

**Cut:**
- Figure 14 (linguistic-distance bar chart) — duplicates Figure 13.
- Figure 15 (conceptual surface-vs-deep diagram) — decorative, no new information; delete.
- The "Variety-Specific Slang Markers" paragraph if you kept the slang figure in §1.1; otherwise keep it here.

**Estimated saving:** ~0.3 pages.

---

## Section 2 — lose 4 pages

This is where the biggest cuts live. Three structural changes:

### Structural change A — Move the canonical "class imbalance + Macro-F1 rationale" paragraph to §1.1 once

Right now class imbalance is restated four times: §1.1, §2.1 (during RoBERTa intro), §2.2 (with exact numbers), §2.3 (LoRA section). **Keep the version in §1.1** and replace the others with one-line backreferences like *"as discussed in §1.1, we use weighted cross-entropy throughout"*.

**Estimated saving across §2:** ~0.4 pages.

### Structural change B — Collapse SVM into a 2-sentence aside

Replace the entire **A2: SVM + TF-IDF** subsection in §2.1 (lines 45–48 of the docx, roughly half a page) with two sentences inside the LR paragraph:

> *We also fitted LinearSVC on the same TF-IDF features as a robustness check. Results are within 0.01 Macro-F1 of LR on both tasks (Sentiment 0.821, Sarcasm 0.623) and identical at SD = 0.000 across both seeds, confirming that the binding constraint is the TF-IDF representation, not the choice of classifier (full per-class metrics in Table 5).*

**Estimated saving:** ~0.4 pages here (and another ~0.5 in §3.2 — see below).

### Structural change C — Cut the "Sarcastic Class Gap" sub-discussion in §2.1

Lines 60–76 in the docx. This section repeats analysis already made earlier in §2.1 (per-variety RoBERTa-vs-baseline, why TF-IDF misses sarcasm, gap-by-variety). Collapse into the single "Comparison" paragraph below.

**Estimated saving:** ~0.7 pages.

### Concrete rewrite — paste-ready §2.1

Replace **everything currently between the §2.1 heading and the §2.2 heading** with:

> ## 2.1 Baseline / PTLM gap (10 marks)
>
> **Classical baseline.** TF-IDF transforms text into numerical features by weighting words by document frequency relative to the corpus, ignoring word order and context. We fit two task-specific classifiers — Logistic Regression and LinearSVC — on identical TF-IDF features (15,000 features, unigrams + bigrams) with `class_weight='balanced'` and `max_iter=2000`, evaluated on the all-pooled test set across seeds 42 and 123. TF-IDF features are deterministic so SD = 0.000. Results are within 0.01 Macro-F1 of each other on both tasks, confirming that the binding constraint is the TF-IDF representation, not the choice of classifier; for the rest of the report we use LR as the canonical classical baseline (full per-class metrics for both classifiers in Tables 2 and 5).
>
> **PTLM baseline.** RoBERTa-base is a bi-directional encoder; every token attends to every other token, capturing the full-sentence context that sarcasm requires. For this comparison we fine-tuned RoBERTa-base on the all-pooled training set with weighted cross-entropy (class weights as in §1.1), `lr=1e-5`, `epochs=5`, `warmup=0.1`, `weight_decay=0.01`, averaged across seeds 42 and 123.
>
> *[Table 2 — Sentiment Macro-F1, Sarcasm Macro-F1, Sentiment Acc, Sarcasm Acc for {LR, SVM, RoBERTa-base}.]*
>
> **Gap.** Fine-tuned RoBERTa beats both classical baselines by ~0.10 Macro-F1 on sentiment and ~0.15 on sarcasm. The Non-Sarcastic-F1 is essentially identical across the three models; the gap is concentrated entirely on the Sarcastic-F1 (TF-IDF-LR 0.27 → RoBERTa 0.46), confirming that the contextual representation, not the classifier, drives the improvement (Skalicky & Crossley 2018). The PTLM advantage is largest on en-UK (+0.42 Macro-F1) and en-IN (+0.13), where sarcastic training data is scarcest, and smallest on en-AU (+0.19), consistent with pre-training compensating most when task-specific data is scarce (Devlin et al. 2019). The same per-variety ranking — en-AU > en-UK > en-IN — appears in both classical and transformer models, indicating that variety difficulty is a property of the data rather than of any particular model family.
>
> *[Figure 21 — single 1×2 panel: (left) Macro-F1 by model and task; (right) Sarcastic-F1 by variety, classical vs RoBERTa.]*

**Old: ~3 pages. New: ~1.4 pages. Saving: ~1.6 pages.**

### 2.2 RoBERTa cross-variety — modest trim

**Keep:** the asymmetry analysis (inner vs outer circle), the all-pool consistency claim, the variety-gap of 0.135 finding, the "IN→UK > UK→IN" insight, Figure 25.

**Cut:**
- The class-imbalance recap paragraph (the one that re-quotes 29.4% / 7.63% / 6.81% — already in §1.1).
- The standalone "UK-only instability (std=0.045) ... " paragraph and the "IN-only showed the most dramatic improvement ..." paragraph — collapse both into one sentence each inside the main analysis.

### Concrete rewrite — §2.2 trim

Replace the docx paragraphs from "We extended the initial requirements ..." through "...directly motivating the LoRA variety-specific adaptation in Section 2.3." with the version below. Keep Figure 25.

> ## 2.2 Cross-variety evaluation — RoBERTa (15 marks)
>
> We extend the cross-variety protocol to **five training conditions** — `uk_only`, `au_only`, `in_only`, `inner_pool` (UK+AU), and `all` — each evaluated on every variety's test set, to answer three questions: does the variety gap exist, is it asymmetric between inner- and outer-circle varieties, and does pooling close the gap? RoBERTa-base is used throughout with the weighted-loss setup of §2.1; train/test splits are strictly variety-separated.
>
> *[Figure 25 — 5×3 cross-variety Macro-F1 matrix, mean over seeds 42, 123.]*
>
> **The gap exists and is asymmetric.** `au_only` is the strongest single-variety model (0.760 on the en-AU test set), almost entirely because en-AU has 4× more sarcastic training examples (29.4%) than the other two; this is a class-balance effect, not a "Australian sarcasm is easier" effect. The all-pool condition is the most stable across varieties (UK 0.735, AU 0.744, IN 0.609, σ ≤ 0.015) but does not beat `au_only` on the en-AU test (0.754 vs 0.760), revealing the trade-off between in-variety specialisation when training data is balanced and cross-variety stability for deployment. The inner-circle pool (UK+AU) closes the gap on inner-circle test sets (UK 0.747, AU 0.672) but loses 0.05 on en-IN compared with the all-pool, so adding outer-circle data to training is worth the small inner-circle cost.
>
> **Inner-circle / outer-circle asymmetry.** UK ↔ AU transfer is consistently better than UK ↔ IN or AU ↔ IN (e.g. AU→UK 0.602, AU→IN 0.496), confirming the geographic and historical proximity of inner-circle varieties is reflected in shared sarcastic conventions. A second, less expected asymmetry emerges in the *direction* of outer-circle transfer: IN→UK (0.597) outperforms UK→IN (0.527), and IN→AU (0.579) outperforms AU→IN (0.496). Indian-English-trained models partially generalise upward to inner-circle varieties more than the reverse — likely a consequence of British-English exposure through formal education and media in the en-IN training data.
>
> **Stability.** `uk_only` is unstable across seeds (σ = 0.045) because there are only ~92 sarcastic UK training examples; `in_only` becomes the most stable model in the experiment (σ = 0.0004) once we apply weighted loss and an adjusted learning rate, going from collapse (Macro-F1 0.482 unweighted) to 0.630.
>
> The persistent 0.135 Macro-F1 gap between the best and worst test sets under `all` shows that pooling alone cannot close the en-IN ceiling, motivating the variety-specific adapters in §2.3.

**Old: ~3 pages. New: ~1.4 pages. Saving: ~1.6 pages.**

### 2.3 LoRA — cut intro and three-studies prose

**Keep:** the LoRA equation, the OPT-1.3B base motivation, the ablation table (Table 4), the final-config sentence.

**Cut:**
- The "What LoRA is and why we use it" intro paragraph + the three-bullet "this matters because..." block — replace with two sentences.
- The full "Class imbalance" subsection (it's the 4th time the report explains class weights) — replace with a one-line backreference.
- The "Three studies" header + descriptions of Study 1/2/3 (we just need to *show* the studies, not narrate them).
- The hyperparameter ablation prose. Keep Table 4, drop the paragraph that re-states the table values.

### Concrete rewrite — §2.3 trim

Replace **everything from the §2.3 heading through the end of "Hyperparameter ablation (Study 1)"** in the docx with:

> ## 2.3 LoRA adapters (15 marks)
>
> LoRA (Hu et al. 2021) freezes the base model's weights and learns two small low-rank matrices A ∈ ℝ^(d×r), B ∈ ℝ^(r×d) attached to the attention projections, such that `output = W_frozen·x + (α/r)·A·B·x`. With r=8 on OPT-1.3B this trains 1.6 M parameters out of 1.32 B (0.12%) and yields per-variety adapters that are 6 MB each — small enough that one frozen base + three adapters can serve all three varieties from the same machine, swapping in microseconds at inference time (used in §5.1).
>
> **Setup.** Frozen base: `facebook/opt-1.3b` (fp16 weights). Tokenisation: OPT tokenizer, 128-token truncation (95th-percentile Reddit comment length). Class imbalance is handled with weighted cross-entropy as defined in §1.1, implemented in `WeightedTrainer` (`src/functions_to_use.py`).
>
> **Hyperparameter ablation (1 epoch, en-UK).** We grid-searched r ∈ {4, 8, 16}, lr ∈ {1e-4, 2e-4}, weighted ∈ {True, False}.
>
> *[Table 4 — 12 configurations, sorted by Macro-F1.]*
>
> Two patterns matter. (i) `r=4, lr=1e-4` collapses to predicting the majority class for all inputs — adapter capacity is too small at that learning rate to leave the random-init basin. (ii) The three highest Macro-F1 configurations are *unweighted*, but their Sarcastic-F1 is near zero — the appearance of better Macro-F1 is a class-imbalance artefact. We retain class weighting because Sarcastic-F1 is the metric we actually care about and because en-IN's stronger imbalance (7%) would collapse without it. **Final config: r=8, lora_alpha=16, lr=2e-4, weighted=True** (Macro-F1 0.7560 in the ablation), in line with Hu et al. 2021 for ≥1B-parameter models.
>
> **Training and cross-variety evaluation** are reported in §3.4; the per-variety adapters are released at <https://huggingface.co/momofahmi>.

**Old: ~2 pages. New: ~0.8 pages. Saving: ~1.2 pages.**

### Total §2 trim: ~4.4 pages saved ✓

---

## Section 3 — lose 3 pages

### 3.1 LR + TF-IDF — modest trim

**Keep:** the Macro-F1 rationale paragraph (this is the right place for it — keep here, then in §1.1 just reference §3 for the metric), confusion matrix Fig 26, per-class P/R/F1 Fig 27.

**Cut:**
- The first paragraph ("Macro-F1 is used as ...") if you've already moved this rationale to §1.1; keep one sentence.
- The "lazy model would yield 0.0 recall" sentence — implied by the table.

**Estimated saving:** ~0.3 pages.

### 3.2 SVM + TF-IDF — DELETE the entire subsection

This whole subsection currently re-states results that are functionally identical to §3.1. Replace it with a single line at the end of §3.1:

> *LinearSVC with the same features and class weighting yields Macro-F1 0.821 (sentiment) / 0.623 (sarcasm), within 0.01 of LR; per-class numbers are in Table 5.*

Move Table 5 (SVM per-class) to the appendix or merge with Table 2 in §2.1.

**Estimated saving:** ~0.7 pages.

### 3.3 RoBERTa — modest trim

**Keep:** Figure 31 (cross-variety matrix), Figure 33 (best-condition confusion matrix), the per-class F1 paragraph that proves the model isn't collapsing.

**Cut:**
- Figure 32 (per-class F1 across all conditions) — duplicates information already in Figure 31.
- The class-imbalance / std-discussion paragraphs that re-state §2.2 — collapse into one sentence: *"Per-condition stability and per-class F1 are reported in §2.2."*

**Estimated saving:** ~0.6 pages.

### 3.4 LoRA — heavy cut

This subsection is currently ~2 pages and re-states most of §2.3. Cut almost everything except the cross-variety tables and the BESSTIE comparison.

**Cut entirely:**
- "Metrics" subsection — already explained in §3 intro; one line of context is enough.
- "Training dynamics (Study 2)" subsection — *exactly the same content as §2.3 Study 2*; delete.
- "Effect of the frozen base" prose — keep Table 8 but drop the discussion paragraphs (they say "OPT > LLaMA on all varieties because pretraining data, increasing rank doesn't help" — say it in two sentences).
- "Summary and Discussion" subsection — the deployment-decision paragraph belongs in §5; the "we expected larger model would help, it didn't" paragraph belongs in §2.3 if anywhere.

**Keep:** Tables 6 and 7 (cross-variety Macro-F1 and Sarcasm-F1), Table 8 (frozen-base comparison), Figure 33 (LoRA training curves) — though if §2.3 already shows curves, drop here and reference §2.3.

### Concrete rewrite — §3.4

Replace **the entire §3.4 subsection** with:

> ### 3.4 LoRA — best-adapter, in-variety, cross-variety
>
> Cross-variety performance of the three OPT-1.3B + LoRA adapters trained in §2.3, evaluated on each of the three test sets, two seeds per cell (18 prediction runs total).
>
> *[Table 6 — Macro-F1 (mean ± SD over seeds 42, 123). Bold = in-variety diagonal.]*
>
> *[Table 7 — Sarcastic-class F1 (minority class), mean over seeds.]*
>
> The Macro-F1 matrix is partly inflated by the easy non-sarcastic majority — for example en-IN-adapter scores 0.758 on the UK test set, slightly higher than the en-UK adapter (0.751), but with σ = 0.041 the result is not stable across seeds. The Sarcastic-F1 matrix is cleaner. The en-AU adapter drops from 0.68 in-variety to 0.26–0.36 cross-variety (≈ 0.4 F1 points), the en-IN adapter generalises *upward* to en-UK (0.56) better than the en-UK adapter generalises *downward* to en-IN (0.37) — the same asymmetry seen with RoBERTa in §2.2.
>
> **Comparison with the BESSTIE baseline (Srirag et al. 2025, MISTRAL 22B).** Their best-decoder Sarcastic-F1 is 0.71 / 0.68 / 0.44 (UK / AU / IN); ours is 0.54 / 0.68 / 0.39, average 0.54 vs 0.61. We match on en-AU exactly, lose 0.17 on en-UK and 0.05 on en-IN, with a model 17× smaller and only 0.12% of parameters trained.
>
> **Effect of the frozen base.** We re-ran Study 2 with two LLaMA bases under the same configuration.
>
> *[Table 8 — In-variety Sarcastic-F1 by frozen base.]*
>
> OPT-1.3B beats both LLaMA models on all three varieties despite being smaller than LLaMA-3.2-3B; the most plausible explanation is OPT's pre-training data overlap with Reddit (BESSTIE's source). Increasing the LoRA rank from r=4 to r=8 on LLaMA-3.2-3B did not move the metric, suggesting the bottleneck is base-model / domain match rather than adapter capacity. We use OPT-1.3B as the canonical model in §4 (error analysis) and §5 (deployment).

**Old: ~2 pages. New: ~0.7 pages. Saving: ~1.3 pages.**

### Total §3 trim: ~2.9 pages saved ✓

---

## Final budget after these trims

| Section | Before | After | Δ |
|---|---:|---:|---:|
| §1 | 5 | 4.0 | −1.0 |
| §2 | 10 | 5.6 | −4.4 |
| §3 | 8 | 5.1 | −2.9 |
| §4 | 3 | 3 | 0 |
| §5 | 3 | 3 | 0 |
| **Total** | **29** | **20.7** | **−8.3** |

That gives ~4 pages of slack — enough to *grow* §4 (currently 1 page under budget) if you want a richer error analysis discussion, or to keep a couple of figures we tagged for cuts.

---

## Order to apply these trims

If you're editing the docx live, do them in this order to minimise re-flow churn:

1. **§3.4 first** (biggest single cut, ~1.3 pages, contained in one subsection — easiest).
2. **§2.1 SVM collapse** (delete A2 subsection, replace with two sentences).
3. **§3.2 SVM** delete.
4. **§2.1 Sarcastic Class Gap** subsection delete (paragraphs after "COMPARISON").
5. **§2.3 intro and Three Studies** rewrite.
6. **§2.2 modest trim** (drop class-imbalance recap, condense stability paragraphs).
7. **§1 figure consolidation** if still over 25.
8. Run final page count after each step — you'll likely hit 25 before step 7.
