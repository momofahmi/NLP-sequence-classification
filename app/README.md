# Running the BESSTIE deployment locally

This is Mohamed's Gradio app — three OPT-1.3B + LoRA adapters, switchable per English variety.

## Prerequisites

- Python 3.10–3.11 (`python3 --version`)
- ~5 GB free disk for the OPT-1.3B base model + transformer caches
- Internet on first run (pulls weights from HuggingFace Hub)
- **No HF token required** — both the base model (`facebook/opt-1.3b`) and the adapters (`momofahmi/besstie-lora-en-{uk,au,in}-opt-1.3b`) are public.

## One-time setup

From the repo root:

```bash
# 1. Make sure you're on a branch that has the app (origin/main does).
git fetch origin
git checkout main      # or: git checkout fiyin/model-pipeline if you've cherry-picked the app

# 2. Create a virtualenv (recommended).
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies.
pip install -r requirements.txt
```

If `pip install -r requirements.txt` fails on `torch` (Apple Silicon sometimes resolves to a CPU-only wheel that crashes on `from_pretrained`), install torch first explicitly:

```bash
pip install --upgrade pip
pip install "torch>=2.1"          # picks the right wheel for your platform
pip install -r requirements.txt
```

## Run the app

```bash
python app/app.py
```

What you'll see:

```text
Loading base model on cpu...
Loading adapters...
Ready.
* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.
```

Open `http://127.0.0.1:7860` in your browser.

> **First run takes ~30–90 seconds on CPU** because the base model has to download (~2.6 GB) and load. Subsequent runs are fast (cached under `~/.cache/huggingface/`).

> **Inference latency on CPU** (Mac M1/M2 / Intel laptop) is ~2–5 seconds per sentence in single mode and ~5–15 seconds per row in the "Compare all adapters" tab (because it runs three forward passes per sentence). On a Colab T4 GPU it's <200 ms.

## What the two tabs do

- **Sarcasm Detection** — type a sentence, pick a variety (`en-UK`, `en-AU`, `en-IN`), press Analyze. The app calls `peft_model.set_adapter(variety)` and returns `Sarcastic` or `Not Sarcastic` with confidence.
- **Compare all adapters** — paste several sentences (one per line). Each is run through all three adapters and the table shows per-variety predictions plus a verdict (`All sarcastic`, `None is sarcastic`, or `N of 3 sarcastic`). This is what you screenshot for §5.1 of the report — it visually demonstrates the cross-variety disagreement we measured numerically in §2.3.

## Quick smoke-test sentences

Paste these into the **Compare all adapters** tab to see the three adapters disagree:

```
Absolute legend, parked his ute right across my driveway. Good onya, mate.
Coz we all have free internet.
Cheerful fellow aren't you.
What a brave potatriot
The Interior was Too Good and Test Was Awesome.
```

Expected behaviour:
- Line 1 (en-AU sarcasm) — `en-AU` adapter most likely flags it as Sarcastic.
- Line 2 (en-IN sarcasm) — `en-IN` adapter most likely flags it as Sarcastic; the others may miss it.
- Line 3 (en-UK sarcasm) — `en-UK` adapter most likely flags it as Sarcastic.
- Line 4 (en-UK political sarcasm) — interesting: depends heavily on the adapter.
- Line 5 (en-IN positive review, not sarcastic) — all three should agree it's Not Sarcastic.

## Capture the screenshots for §5.1

The deployment write-up in `reports/results/q5_1_deployment.md` references three screenshots:

- **`[FIGURE 5.1.1]`** — system diagram (already described in prose, you can draw this in Google Slides if you want a visual; otherwise omit).
- **`[FIGURE 5.1.2]`** — single-sentence prediction in tab 1 with a sarcastic en-AU input.
- **`[FIGURE 5.1.3]`** — the "Compare all adapters" tab with the smoke-test sentences above showing cross-variety disagreement.

Screenshot tool on macOS: `Cmd+Shift+4` then drag to capture a region. Save into `reports/figures/q5_1_screenshot_*.png` and reference in the docx.

## Stopping the app

`Ctrl+C` in the terminal.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'gradio'` | Activate your venv (`source .venv/bin/activate`) and re-run `pip install -r requirements.txt`. |
| `ConnectionError` on first launch | Your network blocks `huggingface.co`. Either retry or set `HF_HUB_OFFLINE=0` and try a VPN. |
| `RuntimeError: Some tensors share memory…` | Old `accelerate` version. `pip install -U accelerate`. |
| Browser opens but page is blank | Gradio sometimes needs `--server-name 0.0.0.0`. Edit the last line of `app/app.py` to `demo.launch(server_name="0.0.0.0")`. |
| Out of RAM on a 16 GB MacBook | OPT-1.3B in fp32 needs ~6 GB. Close other apps; if still OOM, edit `app/app.py` and force `dtype = torch.float16` on CPU too (slightly less stable but usually works). |
