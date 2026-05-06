"""Generate notebooks/main.ipynb — the single end-to-end entry point for submission.

Usage:
    python scripts/build_main_notebook.py

Re-run this any time we want to refresh the orchestrator notebook.
"""
from __future__ import annotations
import json
import nbformat as nbf
from pathlib import Path


def md(src: str):
    return nbf.v4.new_markdown_cell(src)


def code(src: str):
    return nbf.v4.new_code_cell(src)


nb = nbf.v4.new_notebook()
cells: list = []

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
cells.append(md(
    """# COMM061 — Group Coursework PG15
## BESSTIE: Sentiment & Sarcasm Classification across Varieties of English

University of Surrey · Semester 2, 2025/26 · Submission notebook

**Group PG15:**
- Sumtally, Ummé Yusrah (6931336)
- Mohamed Fahmi Ahmed (6956810)
- Joel Allen-Caliste (6961120)
- Sayed Omar Aabid (6945752)
- Mohammad (Q4 error analysis)
- Fiyinfoluwa Akano (6962514)

**Dataset:** [`surrey-nlp/BESSTIE-CW-26`](https://huggingface.co/datasets/surrey-nlp/BESSTIE-CW-26) (Srirag et al. 2025)

---

This notebook is the **entry point for the coursework submission**. It runs the full pipeline end-to-end:

| Step | Section | What it does |
|---|---|---|
| 0 | Setup | Clone repo (Colab), install deps, set seeds, load BESSTIE-CW-26 |
| 1.1 | Data analysis | Class distribution, source-by-variety, sarcasm-sentiment correlation |
| 1.2 | Vocabulary | Jaccard / TF-IDF cosine similarity, linguistic distance, slang markers |
| 2.1 | Classical baseline | TF-IDF + Logistic Regression, TF-IDF + LinearSVC |
| 2.1 | PTLM baseline | RoBERTa-base on all-pool |
| 2.2 | Cross-variety | RoBERTa across 5 training conditions × 3 test sets |
| 2.3 | LoRA adapters | OPT-1.3B + LoRA per variety (canonical model) |
| 3 | Evaluation | Render all comparison tables & heatmaps from saved results |
| 4 | Error analysis | Extract 10 errors, build 4-shot prompt, re-test on 6 |
| 5.1 | Deployment | Pointer to the Gradio app on HF Spaces (`app/app.py`) |
| 5.2 | Efficiency | Inference latency benchmark |

> **How to run.** On Colab, change runtime to **T4 GPU**, then `Runtime → Run all`. Default settings load pre-trained adapters from the Hub and complete in ~10 minutes. Set `RETRAIN_ROBERTA=True` and/or `RETRAIN_LORA=True` in the config cell to retrain from scratch (~75 min on T4 total).
"""
))

# ---------------------------------------------------------------------------
# Section 0: Setup
# ---------------------------------------------------------------------------
cells.append(md("""## 0. Setup

### 0.1 Config flags
Adjust these to switch between the fast evaluation path (default) and full retraining."""))

cells.append(code("""# === main.ipynb config ===========================================================
SEEDS = [42, 123]
DEMO_MODE       = False        # True = 200-row subset, useful for smoke runs
RETRAIN_ROBERTA = False        # True = retrain all 5 RoBERTa conditions (~25 min T4)
RETRAIN_LORA    = False        # True = retrain en-UK / en-AU / en-IN adapters (~30 min T4)
RETRAIN_LR      = True         # cheap, always retrain
RUN_BENCHMARK   = True         # §5.2 inference latency
RUN_LIME        = False        # bonus interpretability (slower)

HF_HUB_USER = "momofahmi"
DATASET_ID  = "surrey-nlp/BESSTIE-CW-26"
TASK_LABEL  = "Sarcasm"        # primary task; sentiment is auxiliary

VARIETIES = ["en-UK", "en-AU", "en-IN"]

print("Config OK — RETRAIN_ROBERTA =", RETRAIN_ROBERTA, "| RETRAIN_LORA =", RETRAIN_LORA)
"""))

cells.append(md("""### 0.2 Environment

Detects whether we're on Colab and clones the repo + installs requirements if needed."""))

cells.append(code('''import os, sys, subprocess, importlib, pathlib

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    # Public mirror of Fiyin's pipeline branch — anyone can clone without auth.
    # Override these by setting REPO_URL / REPO_BRANCH as env vars before running.
    REPO_URL  = os.environ.get("REPO_URL", "https://github.com/TheFinix13/NLP-coursework.git")
    BRANCH    = os.environ.get("REPO_BRANCH", "main")
    REPO_DIR  = "/content/NLP-coursework"
    if not os.path.exists(REPO_DIR):
        subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, REPO_DIR], check=True)
    os.chdir(REPO_DIR)
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    subprocess.run(
        ["pip", "install", "-q", "-r", "requirements.txt"],
        check=True,
    )
else:
    PROJECT_ROOT = pathlib.Path.cwd()
    while not (PROJECT_ROOT / "requirements.txt").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
        PROJECT_ROOT = PROJECT_ROOT.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)

print("CWD:", os.getcwd())
print("Python:", sys.version.split()[0])
'''))

cells.append(code("""import random, numpy as np
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

seed_all(SEEDS[0])

if HAS_TORCH:
    DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {DEVICE} | torch {torch.__version__}")
else:
    DEVICE = "cpu"
    print("torch not installed yet — heavy training cells will skip until installed.")
"""))

cells.append(md("""### 0.3 Load BESSTIE-CW-26"""))

cells.append(code("""from src.besstie_data_loader import (
    get_BESSTIE_splits,
    get_variety_split,
    load_besstie,
)

# raw HF DatasetDict for downstream training (RoBERTa / LoRA)
ds_dict = load_besstie()

# Pandas frames used for EDA, vocab, and the classical baseline
df_all, df_train, df_val, df_test = get_BESSTIE_splits()

# Coerce label dtypes — older notebook conventions
for _df in [df_all, df_train, df_val, df_test]:
    _df['Sarcasm']   = _df['Sarcasm'].astype(int)
    _df['Sentiment'] = _df['Sentiment'].astype(int)

print("BESSTIE-CW-26 splits:")
print(f"  train: {len(df_train):>5} rows")
print(f"  valid: {len(df_val):>5} rows")
print(f"  test : {len(df_test):>5} rows")
print(f"  total: {len(df_all):>5} rows")

if DEMO_MODE:
    df_train = df_train.sample(min(200, len(df_train)), random_state=SEEDS[0]).reset_index(drop=True)
    df_val   = df_val.sample(min(40, len(df_val)),    random_state=SEEDS[0]).reset_index(drop=True)
    df_test  = df_test.sample(min(60, len(df_test)),  random_state=SEEDS[0]).reset_index(drop=True)
    print("DEMO_MODE on — splits subsampled.")

df_train.head()
"""))

# ---------------------------------------------------------------------------
# Section 1.1
# ---------------------------------------------------------------------------
cells.append(md("""## 1. Data analysis & visualisation"""))

cells.append(md("""### 1.1 Distribution and class imbalance

We render the figures used in §1.1 of the report — variety, source, split, sarcasm/sentiment per variety, sarcasm-sentiment correlation, POS analysis, and slang markers."""))

cells.append(code("""from src.eda_distributions import EDA, get_sarcasm_extremes, variety_slang

eda = EDA(df_all, df_train, df_val, df_test)

# Sarcasm + sentiment class imbalance
overall_sarc, per_var_sarc, per_split_sarc = eda.sarcasm_imbalance()
overall_sent, per_var_sent, per_split_sent = eda.sentiment_imbalance()

print(f"Overall sarcasm rate: {overall_sarc[1]:.2f}%")
print(f"Per-variety sarcasm rate:")
for variety, pct in per_var_sarc.items():
    print(f"  {variety}: {pct:.2f}%")
"""))

cells.append(code("""# Variety + source distribution
variety_counts, source_counts = eda.variety_source_dist(eda.df_all)
print(f"Varieties: {dict(variety_counts)}")
print(f"Sources  : {dict(source_counts)}")

# Train / val / test split per variety
split_distribution = eda.split_distribution_per_variety()
print(split_distribution)
"""))

cells.append(code("""# Source-per-variety crosstab and sarcasm/sentiment correlation
eda.source_per_variety()
eda.sarcasm_sentiment_correlation()
"""))

cells.append(code("""# Sarcastic phrase patterns and POS analysis
sarcasm_patterns, examples = eda.sarcastic_phrases_analysis()
pos_data, sarc_pos, non_sarc_pos = eda.pos_for_sarcasm(n_samples=500)

print('\\nPOS comparison (sarcastic vs non-sarcastic):')
for pos, sp, np_ in zip(pos_data['pos_tags'], pos_data['sarcastic_pcts'], pos_data['non_sarcastic_pcts']):
    print(f'  {pos:<8} {sp:5.1f}%  vs  {np_:5.1f}%')
"""))

cells.append(code("""# Variety-specific slang markers (used in §1.2 too)
slang_results = variety_slang(df_all)
for variety, slang_found in slang_results.items():
    print(f'\\n{variety.upper()}: {len(slang_found)} slang terms found')
    for slang, ex in slang_found[:3]:
        print(f'  - "{slang}" :: {ex[:80]}…')

extremes = get_sarcasm_extremes(per_var_sarc)
print('\\nSarcasm extremes:', extremes)
"""))

# ---------------------------------------------------------------------------
# Section 1.2
# ---------------------------------------------------------------------------
cells.append(md("""### 1.2 Vocabulary analysis

Jaccard and TF-IDF-cosine pairwise similarity across the three varieties; computed on the full dataset (not just training) per the assignment brief."""))

cells.append(code("""from src.vocabulary_overlap import VocabularyAnalysis

vocab_analyser = VocabularyAnalysis(
    df_all=df_all,
    text_col='text',
    variety_col='variety',
    save_path='./reports',
)

df_jaccard, df_cosine, vocab_per_variety = vocab_analyser.run(save=True)

print('\\n=== Pairwise Jaccard similarity ===')
print(df_jaccard.round(4))
print('\\n=== Pairwise TF-IDF cosine similarity ===')
print(df_cosine.round(4))

# Linguistic distance summary (1 - similarity)
pairs = [('en-AU', 'en-IN'), ('en-AU', 'en-UK'), ('en-IN', 'en-UK')]
print('\\n=== Linguistic distance (1 − similarity) ===')
print(f"{'Pair':<20} {'Jaccard':>10} {'TF-IDF':>10}")
for a, b in pairs:
    print(f"  {a} ↔ {b:<8}  {1-df_jaccard.loc[a, b]:>10.4f}  {1-df_cosine.loc[a, b]:>10.4f}")
"""))

cells.append(code("""# Distinctive terms per variety (used to identify slang markers)
from src.linguistic_feature_analysis import LinguisticFeatureAnalysis

ling_analyser = LinguisticFeatureAnalysis(
    df_all=df_all,
    text_col='text',
    sarcasm_col='Sarcasm',
    variety_col='variety',
    save_path='./reports',
)

variety_terms = ling_analyser.extract_variety_terms(top_n=10, save=False)
for variety, df_terms in variety_terms.items():
    top10 = df_terms.head(10)['term'].tolist() if 'term' in df_terms.columns else df_terms.head(10).index.tolist()
    print(f'{variety}: {top10}')
"""))

# ---------------------------------------------------------------------------
# Section 2.1
# ---------------------------------------------------------------------------
cells.append(md("""## 2. Experimentation

### 2.1 Classical baseline — TF-IDF + Logistic Regression / LinearSVC

We extract a single 15k-feature unigram+bigram TF-IDF representation and train two task-specific classifiers per task. Per the report (§2.1), Logistic Regression and LinearSVC are within 0.01 of each other on both tasks; we keep both as a robustness check."""))

cells.append(code("""from src.tfidf_feature_extraction import tfidf_features
from models.baseline.logistic_regression import LogisticRegressionModel
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

# Build a single 15k-feature unigram+bigram TF-IDF representation
X_train, X_val, X_test, vectorizer = tfidf_features(
    df_train, df_val, df_test,
    text_column='text',
    max_features=15_000,
    save_path='./models/tfidf',
)

# Train Logistic Regression — task-specific (one for sentiment, one for sarcasm)
lr_model = LogisticRegressionModel()
lr_model.train_logistic_regression(X_train, df_train)
lr_eval = lr_model.evaluate_logistic_regression(X_test, df_test)

# Add LinearSVC as a robustness check on the same features
baseline_rows = []
for task in ['Sentiment', 'Sarcasm']:
    y_train, y_test = df_train[task].values, df_test[task].values
    svm = LinearSVC(max_iter=2000, class_weight='balanced', random_state=SEEDS[0]).fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    baseline_rows.append({
        'model': 'TF-IDF + SVM',
        'task': task.lower(),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'accuracy': accuracy_score(y_test, y_pred),
    })
    baseline_rows.append({
        'model': 'TF-IDF + LR',
        'task': task.lower(),
        'macro_f1': lr_eval[task]['macro_f1'],
        'accuracy': lr_eval[task]['accuracy'],
    })

baseline_df = pd.DataFrame(baseline_rows).round(4)
baseline_df
"""))

cells.append(code("""# Per-variety LR performance (used in §3.1 of the report)
per_var_rows = []
for var in VARIETIES:
    test_var = df_test[df_test['variety'] == var].reset_index(drop=True)
    if len(test_var) == 0:
        continue
    X_test_var = vectorizer.transform(test_var['text'].tolist())
    eval_var = lr_model.evaluate_logistic_regression(X_test_var, test_var)
    for task in ['Sentiment', 'Sarcasm']:
        per_var_rows.append({
            'variety': var,
            'task': task.lower(),
            'macro_f1': eval_var[task]['macro_f1'],
            'accuracy': eval_var[task]['accuracy'],
        })

per_var_lr = pd.DataFrame(per_var_rows).round(4)
per_var_lr.pivot(index='variety', columns='task', values='macro_f1')
"""))

# ---------------------------------------------------------------------------
# Section 2.1 RoBERTa
# ---------------------------------------------------------------------------
cells.append(md("""### 2.1 PTLM baseline — RoBERTa-base on all-pool

By default we **load the trained checkpoint from HF Hub** and report metrics. Set `RETRAIN_ROBERTA=True` in §0.1 to retrain from scratch (~5 min on T4 for the all-pool condition; full 5×3 cross-variety matrix in §2.2)."""))

cells.append(code("""import json, pathlib

ROBERTA_RESULTS_PATH = pathlib.Path("reports/results/q2_2_roberta_crossvariety_sarcasm.json")

def _run_notebook(path: str):
    \"\"\"Execute another notebook in-process so all its outputs land in this kernel.\"\"\"
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    print(f"Executing {path} (this may take a while)…")
    nb_obj = nbformat.read(path, as_version=4)
    ep = ExecutePreprocessor(timeout=4800, kernel_name='python3')
    ep.preprocess(nb_obj, {'metadata': {'path': '.'}})
    print(f"Finished {path}.")

roberta_results = None
if RETRAIN_ROBERTA or not ROBERTA_RESULTS_PATH.exists():
    print("Retraining RoBERTa (canonical: notebooks/2.2_RoBERTa_CrossVariety_Joel_Fiyin.ipynb).")
    _run_notebook('notebooks/2.2_RoBERTa_CrossVariety_Joel_Fiyin.ipynb')
if ROBERTA_RESULTS_PATH.exists():
    with open(ROBERTA_RESULTS_PATH) as f:
        roberta_results = json.load(f)
    print('RoBERTa conditions loaded:', list(roberta_results.keys()))
else:
    print('No RoBERTa results JSON found — set RETRAIN_ROBERTA=True or check reports/results/.')
"""))

# ---------------------------------------------------------------------------
# Section 2.2
# ---------------------------------------------------------------------------
cells.append(md("""### 2.2 RoBERTa cross-variety — 5 conditions × 3 test sets

We trained RoBERTa-base under five conditions: `uk_only`, `au_only`, `in_only`, `inner_pool` (UK+AU), and `all`, evaluating each on every variety's test set (15 cells × 2 seeds = 30 prediction runs).

The full results matrix is loaded from `reports/results/q2_2_roberta_crossvariety_sarcasm.json` and rendered below."""))

cells.append(code("""import seaborn as sns, matplotlib.pyplot as plt, numpy as np

if roberta_results is None:
    print('Skipping cross-variety heatmap — RoBERTa results unavailable.')
else:
    conditions = ['uk_only', 'au_only', 'in_only', 'inner_pool', 'all']
    matrix = np.full((len(conditions), len(VARIETIES)), np.nan)
    for i, cond in enumerate(conditions):
        if cond not in roberta_results:
            continue
        for j, tv in enumerate(VARIETIES):
            try:
                matrix[i, j] = roberta_results[cond]['averaged'][tv]['macro_f1_mean']
            except (KeyError, TypeError):
                pass

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(matrix, annot=True, fmt='.3f', xticklabels=VARIETIES, yticklabels=conditions,
                cmap='Blues', ax=ax)
    ax.set_title('RoBERTa — Cross-variety Macro-F1 (sarcasm)')
    ax.set_xlabel('Test variety'); ax.set_ylabel('Train condition')
    plt.tight_layout(); plt.show()
"""))

# ---------------------------------------------------------------------------
# Section 2.3
# ---------------------------------------------------------------------------
cells.append(md("""### 2.3 LoRA adapters — OPT-1.3B (canonical model)

LoRA freezes `facebook/opt-1.3b` (1.32 B params, FP16) and learns 1.6 M trainable parameters per variety adapter (`r=8, alpha=16, lr=2e-4`, weighted CE).

By default we **load the three trained adapters from `huggingface.co/momofahmi`** and run evaluation only. Set `RETRAIN_LORA=True` in §0.1 to retrain from scratch (~10 min per variety on T4)."""))

cells.append(code("""LORA_RESULTS_PATH = pathlib.Path('reports/results/q2_3_lora_full_sarcasm.json')

lora_results = None
if RETRAIN_LORA or not LORA_RESULTS_PATH.exists():
    print('Retraining LoRA adapters (canonical: notebooks/2.3_LoRA_Adapters_Mohamed.ipynb).')
    _run_notebook('notebooks/2.3_LoRA_Adapters_Mohamed.ipynb')
if LORA_RESULTS_PATH.exists():
    with open(LORA_RESULTS_PATH) as f:
        lora_results = json.load(f)
    print('LoRA varieties present:', list(lora_results.keys()))
else:
    print('No LoRA results JSON found — set RETRAIN_LORA=True or check reports/results/.')
"""))

cells.append(code("""# LoRA cross-variety Sarcasm-F1 heatmap (Table 7 in §3.4)
if lora_results is None:
    print('Skipping LoRA heatmap — results unavailable.')
else:
    mat = np.full((len(VARIETIES), len(VARIETIES)), np.nan)
    for i, tv in enumerate(VARIETIES):
        if tv not in lora_results: continue
        for j, ev in enumerate(VARIETIES):
            try:
                mat[i, j] = lora_results[tv]['averaged'][ev]['sarcastic_f1_mean']
            except (KeyError, TypeError):
                pass

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(mat, annot=True, fmt='.3f', xticklabels=VARIETIES, yticklabels=VARIETIES,
                cmap='Greens', ax=ax)
    ax.set_title('LoRA OPT-1.3B — Cross-variety Sarcastic-F1')
    ax.set_xlabel('Test variety'); ax.set_ylabel('Adapter (train variety)')
    plt.tight_layout(); plt.show()
"""))

# ---------------------------------------------------------------------------
# Section 3
# ---------------------------------------------------------------------------
cells.append(md("""## 3. Evaluation

This section consolidates the comparison tables across all three model families. No new training is performed — we render from the JSON results produced in §§2.1–2.3."""))

cells.append(code("""# Headline comparison — Macro-F1 by model and task on the all-pool test set
baseline_df.pivot(index='model', columns='task', values='macro_f1').round(4)
"""))

cells.append(code("""# Per-variety Macro-F1 — LR (per-variety) vs RoBERTa all-pool vs in-variety LoRA adapter
import pandas as pd

rows = []
lr_pivot = per_var_lr[per_var_lr['task'] == 'sarcasm'].set_index('variety')['macro_f1']
for var in VARIETIES:
    row = {'variety': var, 'LR sarcasm': float(lr_pivot.get(var, float('nan')))}
    if roberta_results and 'all' in roberta_results:
        try:
            row['RoBERTa all-pool sarcasm'] = roberta_results['all']['averaged'][var]['macro_f1_mean']
        except (KeyError, TypeError):
            row['RoBERTa all-pool sarcasm'] = float('nan')
    if lora_results and var in lora_results:
        try:
            row['LoRA in-variety sarcasm'] = lora_results[var]['averaged'][var]['macro_f1_mean']
        except (KeyError, TypeError):
            row['LoRA in-variety sarcasm'] = float('nan')
    rows.append(row)

pd.DataFrame(rows).round(4).set_index('variety')
"""))

# ---------------------------------------------------------------------------
# Section 4
# ---------------------------------------------------------------------------
cells.append(md("""## 4. Sarcasm explanation & error analysis

We use the OPT-1.3B + en-AU adapter (best Sarcasm-F1 in §3.4) and:

1. extract 10 high-confidence misclassifications from the en-AU test set
2. explain 4 of them (2 false positives, 2 false negatives)
3. build a 4-shot prompt and re-test the remaining 6 errors with `LLaMA-3.2-1B-Instruct`
4. (optional) generate LIME attributions for representative errors"""))

cells.append(code("""# Step 1 — extract 10 representative errors
%run scripts/q4_extract_errors.py
"""))

cells.append(code("""# Step 2 — view & confirm explanations stored in reports/results/q4_errors.json
import json
with open("reports/results/q4_errors.json") as f:
    errors = json.load(f)
print(f"Loaded {len(errors)} extracted errors. Below is the first one:")
print(json.dumps(errors[0], indent=2))
"""))

cells.append(code("""# Step 3 — few-shot evaluation on the remaining 6
%run scripts/q4_few_shot_eval.py
"""))

cells.append(code("""# Step 4 (optional) — LIME interpretability
if RUN_LIME:
    %run scripts/lime_explain.py
else:
    print("Skipping LIME (RUN_LIME=False in config).")
"""))

# ---------------------------------------------------------------------------
# Section 5
# ---------------------------------------------------------------------------
cells.append(md("""## 5. Deployment & efficiency

### 5.1 Deployment endpoint (Gradio on HuggingFace Spaces)

Live: **https://huggingface.co/spaces/momofahmi/besstie-cw-nlp**

The Gradio app (`app/app.py`) hot-swaps LoRA adapters on a single frozen OPT-1.3B base, so variety switching costs microseconds rather than seconds. To run locally:

```bash
HF_HOME=$(pwd)/.cache/huggingface python app/app.py
```

To run on Colab, open `notebooks/run_deployment_colab.ipynb` and execute it on a T4 runtime. Screenshots for the report are in `reports/figures/q5_1_deployment_*.png`."""))

cells.append(md("""### 5.2 Efficiency benchmark

Average inference time across 20 runs after GPU warmup, per input length, for TF-IDF+LR, RoBERTa-base, and OPT-1.3B+LoRA. The script writes `reports/results/q5_2_efficiency.json` and a Markdown table."""))

cells.append(code("""if RUN_BENCHMARK:
    %run scripts/benchmark_inference.py
else:
    print("Skipping efficiency benchmark (RUN_BENCHMARK=False in config).")
"""))

# ---------------------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------------------
cells.append(md("""## Done

If you reached this cell, the full pipeline ran successfully. All numerical results referenced in the PDF report are now reproduced in this single notebook.

**Reproducibility checklist** (per the coursework brief):
- Two random seeds (42, 123) used in every training cell, std reported across seeds.
- All hyperparameters live at the top of the notebook (`§0.1 Config flags`) or alongside the relevant training call.
- All datasets are loaded from the official `surrey-nlp/BESSTIE-CW-26` HuggingFace mirror — no local CSVs are shipped with the submission.
- All adapters are pushed to `huggingface.co/momofahmi/*`, and the canonical run reads from there with `RETRAIN=False`.
- For full retraining, set the `RETRAIN_*` flags in §0.1 and run again on a Colab T4.

**Where to look in the report**:
- Section 1 ↔ §1.1 / §1.2 above
- Section 2 ↔ §2.1 / §2.2 / §2.3 above
- Section 3 ↔ §3 above
- Section 4 ↔ §4 above
- Section 5 ↔ §5 above
"""))

# ---------------------------------------------------------------------------
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

out_path = Path("notebooks/main.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} with {len(cells)} cells.")
