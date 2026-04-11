# Q1.2 — Vocabulary overlap / linguistic distance

**Source:** `local_run_summary.json` → `vocab_overlap`, plus figure `reports/figures/q1_2_vocabulary_similarity_heatmap.png`.

## Pairwise similarity (from stored run)

### Jaccard similarity (word sets)

| Pair | Jaccard |
|------|--------:|
| en-AU ↔ en-UK | 0.268 |
| en-AU ↔ en-IN | 0.224 |
| en-IN ↔ en-UK | 0.236 |

### TF‑IDF cosine similarity (concatenated variety documents)

| Pair | Cosine |
|------|--------:|
| en-AU ↔ en-UK | **0.887** |
| en-AU ↔ en-IN | 0.748 |
| en-IN ↔ en-UK | 0.803 |

## How to use in the report

- **Jaccard** is strict (exact word overlap) — **lower** absolute values are normal for social text.
- **TF‑IDF cosine** captures **weighted** lexical overlap — **en-AU vs en-UK** is highest here, consistent with “inner-circle” varieties sharing more surface forms than with **en-IN** (often more code-mixing / different sources).
- Tie this to **§2.2 / §2.3**: higher lexical distance pairs are candidates for **worse cross-variety transfer** if models rely on variety-specific cues.
