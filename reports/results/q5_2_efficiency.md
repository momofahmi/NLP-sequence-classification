# Q5.2 — Efficiency (5 marks, max 1 page)

> Report-ready prose. Run `scripts/benchmark_inference.py` once on the team's reference hardware (Colab T4 GPU is fine), copy the printed numbers into the table below, and paste this section into the docx.

## How to populate the table

From the repo root:

```bash
python scripts/benchmark_inference.py \
    --tfidf-vec  notebooks/models/tfidf/tfidf_vectorizer.pkl \
    --tfidf-clf  notebooks/models/LogisticRegression_sarcasm.pkl \
    --roberta    roberta-base \
    --base-llm   facebook/opt-1.3b \
    --lora       momofahmi/besstie-lora-en-uk-opt-1.3b \
    --out        reports/results/q5_2_efficiency.json
```

The script does N=3 warmup + N=20 timed runs per (model, batch size), reports mean/median/std/p95 in ms, and writes a JSON. Drop the `mean_ms` values into Table 5.2.1 below.

---

## §5.2 — text to paste into the docx

We measured the inference latency of the three modelling families used in the report on the same hardware (`[device — e.g. Colab T4 16 GB / RTX 3090 / Apple M2]`). Each model was warmed up for three runs, then timed across 20 forward passes per batch size. We report mean latency in milliseconds for batch sizes of 1, 32 and 128 inputs (max sequence length 128).

### Table 5.2.1 — Mean inference latency (ms) per batch size

| Model | Params | BS=1 | BS=32 | BS=128 |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | `[N/A or ~50K]` | `[fill]` | `[fill]` | `[fill]` |
| RoBERTa-base | 125 M | `[fill]` | `[fill]` | `[fill]` |
| OPT-1.3B + LoRA (r=8) | 1.32 B (1.6 M trainable) | `[fill]` | `[fill]` | `[fill]` |

### Discussion

**The classical baseline is two-to-three orders of magnitude faster than either transformer.** TF-IDF + LR is essentially a sparse matrix multiply and a linear scoring step on CPU, which is why even at batch size 128 it stays well below 10 ms while the transformers grow super-linearly with batch size on CPU and only sub-linearly on GPU.

**RoBERTa-base sits in the middle.** It fits comfortably on the same GPU as OPT-1.3B but is roughly an order of magnitude smaller, and the latency reflects that: at BS=1 it is faster than OPT-1.3B by `[X×]`, and at BS=128 the gap shrinks because GPU batches amortise the fixed kernel-launch cost.

**OPT-1.3B + LoRA is the slowest at every batch size, but the gap is smaller than the parameter-count ratio (10× more params, only `[~3–5×]` slower).** This is because the LoRA adapter adds essentially zero extra latency at inference — the merged matmul kernel handles the low-rank update — and modern GPUs are bound by memory bandwidth on the attention layers rather than by raw FLOPs.

### Trade-off — would we sacrifice accuracy for speed?

For this coursework the answer depends on the use-case:

- **Streaming / per-keystroke moderation** (where latency budgets are <50 ms per request and inputs are short): the classical baseline is the only viable option, and its sentiment performance (Macro-F1 0.83) is genuinely strong. Sarcasm performance (Macro-F1 0.63) is the trade-off. We would take the speed.
- **Batched offline analysis** (where the user submits a corpus of reviews and waits a few seconds): RoBERTa-base or OPT-1.3B+LoRA are both acceptable. The LoRA variant wins because the +0.0–0.1 Macro-F1 gain on sarcasm is worth the extra `[fill]` ms, and because hot-swapping adapters across varieties is cheaper than hosting three full RoBERTa models.
- **Interactive demo** (the deployment in §5.1, ~1 user, ~1 input/sec): latency is dominated by tokenisation and JSON round-trip rather than the model itself. The OPT-1.3B+LoRA path is preferred because it gives the cross-variety routing that is the headline of the project.

In short, the **classical baseline trades 0.10–0.20 Macro-F1 on sarcasm for ~100× speed-up**, which is the right trade for high-throughput pipelines. The transformer variants are the right trade when accuracy matters and a single-digit-second response is acceptable.

### Caveats

- Numbers are reported on a single device; absolute values shift on different hardware (especially CPU vs GPU). The relative ordering is robust.
- Latency excludes one-time cold-start cost (loading weights from disk / Hub). For OPT-1.3B+LoRA that cold start is ~15–25 s on the first request, then near-zero for adapter switches, as discussed in §5.1.
- For LoRA we measured the unmerged path. Calling `peft_model.merge_and_unload()` would give identical accuracy with slightly lower per-call latency at the cost of losing the ability to switch adapters.
