# Q5.1 — Deployment Endpoint (15 marks, max 5 pages)

> Report-ready prose. Paste into the docx under §5.1. Screenshot placeholders are flagged as `[FIGURE …]` — capture them once when running the app locally / on Spaces.

---

## Architecture

We deploy our best-performing model (the OPT-1.3B + LoRA adapters from §2.3) as an interactive web service using **Gradio**. The service is a single Python process that hosts:

- **One frozen base model** — `facebook/opt-1.3b`, loaded once with `AutoModelForSequenceClassification` (`num_labels=2`, `torch.float16` on GPU / `float32` on CPU).
- **Three LoRA adapters**, one per English variety, hot-loaded from the HuggingFace Hub:
  - `momofahmi/besstie-lora-en-uk-opt-1.3b`
  - `momofahmi/besstie-lora-en-au-opt-1.3b`
  - `momofahmi/besstie-lora-en-in-opt-1.3b`

The adapters are attached to the base model via `peft.PeftModel` and are switched at request time with a single call:

```python
peft_model.set_adapter(variety)   # variety ∈ {"en-UK", "en-AU", "en-IN"}
```

This means one model replica in memory serves all three varieties — there is no need to load three full 1.3B models.

`[FIGURE 5.1.1 — System diagram: Browser → Gradio frontend → Tokeniser → Frozen OPT-1.3B body → switchable LoRA adapter → softmax → label]`

---

## User interface

The app exposes two tabs (see `app/app.py`):

**Tab 1 — “Sarcasm Detection”.** A textarea for the input sentence and a radio button to pick the variety (`en-UK`, `en-AU`, `en-IN`). The backend calls `set_adapter(variety)` and returns the label (Sarcastic / Not Sarcastic) plus the softmax confidence.

`[FIGURE 5.1.2 — Screenshot of Tab 1: a sarcastic en-AU input correctly flagged, showing the chosen adapter and confidence]`

**Tab 2 — “Compare all adapters”.** The user pastes a batch of sentences (one per line). Each sentence is run through all three adapters in sequence and the results are returned as a table with a per-row verdict (`All sarcastic`, `None is sarcastic`, or `N of 3 sarcastic`) and a summary line of how many texts produced unanimous vs disagreeing predictions across varieties.

`[FIGURE 5.1.3 — Screenshot of Tab 2: 3–5 sentences with their per-adapter predictions, demonstrating cross-variety disagreement (e.g. text flagged sarcastic by en-AU adapter but missed by en-IN)]`

This second tab directly visualises the cross-variety phenomenon analysed quantitatively in §2.2 / §2.3 — the same input can flip its label depending on which variety the model was fine-tuned on.

---

## Why Gradio

We chose Gradio over Streamlit/Flask for three reasons:

1. **Native HuggingFace integration.** A `gr.Blocks` app can be deployed to HF Spaces with one click, and `peft` / `transformers` are first-class citizens there.
2. **Tab/Block layout fits our two use-cases** (single-prediction vs cross-adapter comparison) without writing routing code.
3. **Lower boilerplate** for asynchronous inference and dataframe outputs than Flask, while keeping the front-end fully Python (no separate JS/HTML).

Streamlit was the alternative; the trade-off is that Streamlit re-runs the whole script on every interaction, which would force us to either re-load the 1.3B base model (~3 s) on every click, or fight `@st.cache_resource`. Gradio keeps the model resident across requests by default. Flask would have given us the most control but required hand-rolling the front-end — not justified for a research demo.

---

## Why hot-swap LoRA adapters instead of three separate models

A naive deployment would load three independent fine-tuned 1.3B models (one per variety). With LoRA we instead keep **one** frozen base and three lightweight adapter files. The savings are:

| Quantity | Three full fine-tunes | OPT-1.3B + 3 LoRA adapters |
|---|---|---|
| Disk per variety | ~2.6 GB (fp16 weights) | **~6 MB** (r=8, target_modules=q_proj,v_proj) |
| GPU memory at serve time | 3 × ~3 GB = ~9 GB | ~3 GB (base) + 3 × 6 MB ≈ **3.02 GB** |
| Switch latency (variety A → B) | Full model reload ≈ 2–3 s | `set_adapter()` ≈ **<1 ms** |
| Cold-start | Three downloads | One download + 3 small adapter pulls |

This is what makes the per-variety routing requirement of the brief practical: a user changing the dropdown does **not** trigger a model reload, just a pointer flip inside `PeftModel`. It also keeps the deployable artefact small enough to host on a free HF Space (CPU tier) for the demo, with the option to swap to a T4 GPU Space for real-time use.

The same architecture would scale trivially to N varieties or to combined sentiment+sarcasm by swapping the adapter set, without re-engineering the service.

---

## Hosting

To avoid bloating the submission ZIP with 1.3 B-parameter weights, the base model is fetched at runtime with `AutoModelForSequenceClassification.from_pretrained("facebook/opt-1.3b")` and the three adapters are pulled directly from the Hub (`PeftModel.from_pretrained(...)` / `load_adapter(...)`). The repository ships only the `app/app.py` source plus the `requirements.txt` line `gradio`, `transformers`, `peft`, `torch`, `accelerate`. The full app is reproducible with:

```bash
pip install -r requirements.txt
python app/app.py     # opens http://127.0.0.1:7860
```

`[FIGURE 5.1.4 — Optional: HF Spaces deployment screenshot showing the live URL]`

---

## Limitations and future work

- **Single task.** The current app routes only sarcasm. Adding sentiment is a one-line addition (a second adapter set + a task radio button).
- **Confidence calibration.** The softmax confidence shown to the user is uncalibrated; for a production deployment we would calibrate with temperature scaling on the validation set.
- **Variety auto-detection.** The user has to manually pick the variety. A natural extension is to plug in a small variety classifier (e.g. fastText on ICE-Corpora as in Srirag et al. 2025) so the app picks the adapter automatically, with a manual override.
