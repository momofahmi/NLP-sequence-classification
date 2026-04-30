# Q1.2 — Vocabulary Analysis (10 marks)

> Report-ready prose. Paste into the docx under §1.2. The numbers come from `local_run_summary.json → vocab_overlap`, and the figure to accompany this section is `notebooks/reports/figures/vocabulary_similarity_heatmap.png`.

---

## §1.2 — text to paste into the docx

We quantify the lexical similarity between the three English varieties in BESSTIE-CW-26 using two complementary measures: **Jaccard similarity** on the raw word-set and **cosine similarity** on TF-IDF vectors built from the variety-specific document concatenations.

Jaccard treats a vocabulary as an unordered set, so a word that appears once contributes the same as one that appears thousands of times — it is a strict measure of *which* words are shared. TF-IDF cosine, in contrast, weights words by how characteristic they are of each variety, so it captures *how much* the shared lexicon dominates everyday usage.

### Table 1.2.1 — Pairwise lexical similarity between English varieties

| Pair | Jaccard | TF-IDF cosine |
|---|---:|---:|
| en-AU ↔ en-UK | 0.268 | **0.887** |
| en-IN ↔ en-UK | 0.236 | 0.803 |
| en-AU ↔ en-IN | 0.224 | 0.748 |

`[FIGURE 1.2.1 — vocabulary_similarity_heatmap.png: 3×3 heatmap of TF-IDF cosine similarity, with the diagonal omitted or set to 1.0]`

### Observations

- The two **inner-circle** varieties — en-AU and en-UK — are the closest pair on both measures (Jaccard 0.268, cosine 0.887). They share British-English orthography, a large stock of common-noun vocabulary, and similar register on Reddit and Google.
- en-IN sits the furthest from en-AU (Jaccard 0.224, cosine 0.748). This is consistent with en-IN being the only **outer-circle** variety in the dataset and the one in which we observed code-mixed Hindi tokens (e.g. *“faltu”*, *“chowkidar”*, *“chori ho jaaega”*) and a markedly different distribution of place-names and brand-names.
- The Jaccard absolute values look low (0.22–0.27), but this is expected for short user-generated text: even comparing two halves of the same variety would not reach 1.0 because rare words appear in only one half. What matters is the **ordering**: en-AU/en-UK > en-IN/en-UK > en-AU/en-IN, and that ordering is identical under TF-IDF cosine.

### Linguistic distance — what does it actually mean?

“Linguistic distance” is the umbrella term for how different two language varieties are. It can be decomposed into at least four orthogonal axes (Joshi et al. 2025):

1. **Lexical distance** — different words for the same concept (en-AU *“arvo”* vs en-UK *“afternoon”*, en-IN *“tempo”* for a small truck).
2. **Orthographic distance** — different spellings of the same word (en-UK *“colour”* vs Indian-English Reddit also using *“color”*).
3. **Grammatical / syntactic distance** — different constructions, e.g. article omission and copula drop in en-IN (*“Kamal is playing cameo role”*), or `there’s` with plural subjects in en-AU (Srirag et al. 2025, eWAVE features).
4. **Pragmatic / cultural distance** — what counts as polite, sarcastic or in-group; this is what makes sarcasm classification harder across varieties even when the words look identical.

Our Jaccard / TF-IDF cosine measures only capture **lexical** distance. They do not see grammatical features (because both rely on bag-of-words tokenisation) and they do not see pragmatic features at all. So the en-AU ↔ en-UK pair scoring 0.887 cosine should not be read as “these varieties are 89% the same” — it means *the distribution of common words is similar*, which is informative but partial.

### Why this matters for the modelling sections (§2.2, §2.3)

If lexical distance were the only thing models had to bridge, we would expect cross-variety transfer to track the table above closely: en-AU↔en-UK transfer should be the easiest, en-AU↔en-IN the hardest. The actual cross-variety matrices in §2.2 (RoBERTa) and §2.3 (LoRA) show that this is *partially* but not fully true:

- For **sentiment classification**, transfer roughly tracks lexical similarity — sentiment-bearing words like *“lovely”*, *“awful”*, *“good”* are shared across varieties, so a model that learns them on en-UK generalises reasonably to en-AU.
- For **sarcasm classification**, cross-variety transfer collapses much harder than lexical similarity alone would predict — the en-AU adapter drops by ~0.4 sarcasm-F1 when tested on en-IN. This is evidence that **sarcasm is driven by pragmatic and cultural distance**, not just lexical distance, and so the high cosine similarity between en-AU and en-UK does not buy back the cross-variety performance one might expect.

This is the link from the data analysis in §1 to the experimental findings in §2.3: the lexical overlap numbers explain the easy half (sentiment generalises), and their *failure* to predict the sarcasm transfer gap motivates everything that follows — variety-specific adapters, weighted losses, and the cross-variety matrix as the headline metric rather than aggregate accuracy.

---

## Source numbers (for reproducibility)

The values in Table 1.2.1 come from `local_run_summary.json` → `vocab_overlap`, which is produced by the EDA notebook (`notebooks/NLP_EDA.ipynb`) and visualised in `notebooks/reports/figures/vocabulary_similarity_heatmap.png`.
