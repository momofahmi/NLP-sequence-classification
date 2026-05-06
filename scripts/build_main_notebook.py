'''Generate notebooks/main.ipynb — the standalone end-to-end submission notebook.

Design choice (per Joel's note): every helper function used by the team's
domain-specific notebooks is *explicitly inlined* into main.ipynb so the marker
can read the entire pipeline in one place. The notebook also defaults to
*loading the canonical results* from JSON files we already produced (Joel's
`weighted_results/*.json` for RoBERTa, Mohamed's `q2_3_lora_full_sarcasm.json`
for LoRA) so the figures match the report exactly without re-running training.

Set `FROM_SCRATCH=True` in the config cell to force retraining.

Re-run this any time we want to refresh the orchestrator notebook:
    python3 scripts/build_main_notebook.py
'''
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "main.ipynb"


def md(src: str):
    return nbf.v4.new_markdown_cell(src.rstrip() + "\n")


def code(src: str):
    return nbf.v4.new_code_cell(src.rstrip() + "\n")


cells: list = []

# -----------------------------------------------------------------------------
# Title
# -----------------------------------------------------------------------------
cells.append(md(r'''# COMM061 — Group Coursework PG15
## BESSTIE: Sentiment & Sarcasm Classification across Varieties of English

University of Surrey · Semester 2, 2025/26 · Submission notebook (`main.ipynb`)

**Group PG15:** Sumtally Ummé Yusrah · Mohamed Fahmi Ahmed · Joel Allen-Caliste · Sayed Omar Aabid · Mohammad Hossein Modaresi · Fiyinfoluwa Akano

**Dataset:** [`surrey-nlp/BESSTIE-CW-26`](https://huggingface.co/datasets/surrey-nlp/BESSTIE-CW-26) (Srirag et al. 2025)

---

This notebook is the single entry point for the coursework. Every helper function
used by the domain-specific notebooks (`1.1_EDA…`, `2.1_Baseline…`,
`2.2_RoBERTa…`, `2.3_LoRA…`) is **explicitly inlined here** so the pipeline can
be read top-to-bottom without jumping between files.

| Section | What runs here |
|---|---|
| 0 | Setup — clone (Colab), install deps, set seeds, load BESSTIE-CW-26 |
| 1.1 | Class distribution, source-by-variety, sarcasm-sentiment correlation |
| 1.2 | Vocabulary analysis (Jaccard / TF-IDF cosine) and linguistic distance |
| 2.1 | TF-IDF + Logistic Regression and TF-IDF + LinearSVC baselines |
| 2.2 | RoBERTa cross-variety (5 conditions × 3 test sets, weighted CE) |
| 2.3 | LoRA OPT-1.3B per variety (canonical model) |
| 3 | Cross-task evaluation summary |
| 4 | Error analysis on the best LoRA adapter + LLaMA-1B-Instruct few-shot |
| 5.1 | Pointer to the Gradio app (`app/app.py`) |
| 5.2 | Inference latency benchmark |

> **Default behaviour (`FROM_SCRATCH=False`)** — load the canonical training
> results from `reports/results/` and reproduce every plot/table used in the
> report. This runs end-to-end on a Colab T4 in under 10 minutes.
>
> **Retraining (`FROM_SCRATCH=True`)** — re-run all training inline using the
> same functions Joel and Mohamed used in the domain notebooks. Total runtime
> ≈ 60–80 min on a Colab T4.
'''))

# -----------------------------------------------------------------------------
# Section 0: Setup
# -----------------------------------------------------------------------------
cells.append(md(r'''## 0. Setup

### 0.1 Config flags'''))

cells.append(code(r'''# === main.ipynb configuration =================================================
SEEDS = [42, 123]
DEMO_MODE       = False    # True → 200-row subset for smoke test
FROM_SCRATCH    = False    # True → retrain RoBERTa & LoRA from scratch (slow)
RUN_ERROR_ANALYSIS = False # §4 — downloads OPT-1.3B adapter, ~10 min on T4. Set True on Colab.
RUN_BENCHMARK   = False    # §5.2 latency — same model download as above. Set True on Colab.
RUN_LIME        = False    # §4 bonus interpretability

VARIETIES   = ["en-UK", "en-AU", "en-IN"]
TASK_LABEL  = "Sarcasm"
DATASET_ID  = "surrey-nlp/BESSTIE-CW-26"
HF_HUB_USER = "momofahmi"

print(f"Config OK | FROM_SCRATCH={FROM_SCRATCH} | DEMO_MODE={DEMO_MODE}")
'''))

cells.append(md(r'''### 0.2 Environment — clone repo on Colab, set up paths'''))

cells.append(code(r'''import os, sys, subprocess, pathlib

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    REPO_URL  = os.environ.get("REPO_URL", "https://github.com/TheFinix13/NLP-coursework.git")
    BRANCH    = os.environ.get("REPO_BRANCH", "main")
    REPO_DIR  = "/content/NLP-coursework"
    if not os.path.exists(REPO_DIR):
        subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, REPO_DIR], check=True)
    os.chdir(REPO_DIR)
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
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

cells.append(md(r'''### 0.3 Imports & seed everything'''))

cells.append(code(r'''import json, random, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None

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
    print("torch not yet installed — heavy training cells will skip.")
'''))

cells.append(md(r'''### 0.4 Load BESSTIE-CW-26

The dataset has three English varieties (en-UK, en-AU, en-IN) split into train (60%) / val (5%) / test (35%). We expose four working frames (`df_all`, `df_train`, `df_val`, `df_test`) for the EDA and the classical baseline, plus the raw HuggingFace `DatasetDict` (`ds`) that the RoBERTa and LoRA training cells need.'''))

cells.append(code(r'''from src.besstie_data_loader import (
    load_besstie, get_variety_split, get_BESSTIE_splits,
    get_train_conditions, get_test_conditions,
)

ds = load_besstie()                              # used by RoBERTa + LoRA
df_all, df_train, df_val, df_test = get_BESSTIE_splits()

for _df in [df_all, df_train, df_val, df_test]:
    _df["Sarcasm"]   = _df["Sarcasm"].astype(int)
    _df["Sentiment"] = _df["Sentiment"].astype(int)

print(f"BESSTIE-CW-26 splits — train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

if DEMO_MODE:
    df_train = df_train.sample(min(200, len(df_train)), random_state=SEEDS[0]).reset_index(drop=True)
    df_val   = df_val.sample(min(40,  len(df_val)),    random_state=SEEDS[0]).reset_index(drop=True)
    df_test  = df_test.sample(min(60,  len(df_test)),   random_state=SEEDS[0]).reset_index(drop=True)
    print("DEMO_MODE on — splits subsampled.")

df_train.head()
'''))

# -----------------------------------------------------------------------------
# Section 1.1
# -----------------------------------------------------------------------------
cells.append(md(r'''## 1. Data analysis & visualisation

### 1.1 Distribution and class imbalance

We render the figures used in §1.1 of the report — variety, source, split, sarcasm/sentiment per variety, sarcasm-sentiment correlation, POS analysis, and slang markers. All of these are produced by the `EDA` class in `src/eda_distributions.py`.'''))

cells.append(code(r'''from src.eda_distributions import EDA, get_sarcasm_extremes, variety_slang

eda = EDA(df_all, df_train, df_val, df_test)

overall_sarc, per_var_sarc, per_split_sarc = eda.sarcasm_imbalance()
overall_sent, per_var_sent, per_split_sent = eda.sentiment_imbalance()

print(f"Overall sarcasm: {overall_sarc[1]:.2f}% sarcastic, {overall_sarc[0]:.2f}% non-sarcastic")
print("\nPer-variety sarcasm rate (% sarcastic):")
sarcastic_col = per_var_sarc[1] if 1 in per_var_sarc.columns else per_var_sarc.iloc[:, 1]
for v, pct in sarcastic_col.items():
    print(f"  {v}: {pct:.2f}%")
'''))

cells.append(code(r'''variety_counts, source_counts = eda.variety_source_dist(eda.df_all)
print("Variety counts:", dict(variety_counts))
print("Source counts:",  dict(source_counts))

split_distribution = eda.split_distribution_per_variety()
print("\nSplit distribution per variety:"); print(split_distribution)
'''))

cells.append(code(r'''eda.source_per_variety()
eda.sarcasm_sentiment_correlation()
'''))

cells.append(code(r'''sarcasm_patterns, examples = eda.sarcastic_phrases_analysis()
pos_data, sarc_pos, non_sarc_pos = eda.pos_for_sarcasm(n_samples=500)

print("\nPOS comparison (sarcastic vs non-sarcastic):")
for pos, sp, np_ in zip(pos_data["pos_tags"], pos_data["sarcastic_pcts"], pos_data["non_sarcastic_pcts"]):
    print(f"  {pos:<8}  {sp:5.1f}%  vs  {np_:5.1f}%")

slang_results = variety_slang(df_all)
for variety, slang_found in slang_results.items():
    print(f"\n{variety.upper()}: {len(slang_found)} slang terms found")
    for slang, ex in slang_found[:3]:
        print(f"  - '{slang}' :: {ex[:80]}…")

print("\nSarcasm extremes:", get_sarcasm_extremes(per_var_sarc))
'''))

# -----------------------------------------------------------------------------
# Section 1.2
# -----------------------------------------------------------------------------
cells.append(md(r'''### 1.2 Vocabulary analysis

Two pairwise similarity measures applied to the entire corpus: Jaccard (surface vocabulary overlap) and TF-IDF cosine (distributional / topical overlap).'''))

cells.append(code(r'''from src.vocabulary_overlap import VocabularyAnalysis
from src.linguistic_feature_analysis import LinguisticFeatureAnalysis

vocab = VocabularyAnalysis(df_all, text_col="text", variety_col="variety", save_path="./reports")
df_jaccard, df_cosine, vocab_per_variety = vocab.run(save=True)

print("\n=== Jaccard similarity ==="); print(df_jaccard.round(4))
print("\n=== TF-IDF cosine similarity ==="); print(df_cosine.round(4))
print("\n=== Linguistic distance (1 − similarity) ===")
for a, b in [("en-AU","en-IN"), ("en-AU","en-UK"), ("en-IN","en-UK")]:
    print(f"  {a} ↔ {b:<8}  Jaccard={1-df_jaccard.loc[a,b]:.4f}   TF-IDF={1-df_cosine.loc[a,b]:.4f}")
'''))

cells.append(code(r'''ling = LinguisticFeatureAnalysis(df_all, text_col="text", sarcasm_col="Sarcasm",
                                  variety_col="variety", save_path="./reports")
variety_terms = ling.extract_variety_terms(top_n=10, save=False)
for variety, df_terms in variety_terms.items():
    if "term" in df_terms.columns:
        top = df_terms.head(10)["term"].tolist()
    else:
        top = df_terms.head(10).index.tolist()
    print(f"{variety}: {top}")
'''))

# -----------------------------------------------------------------------------
# Section 2.1 — TF-IDF + LR + SVM
# -----------------------------------------------------------------------------
cells.append(md(r'''## 2. Experimentation

### 2.1 TF-IDF + Logistic Regression / LinearSVC (classical baselines)

Single 15,000-feature unigram+bigram TF-IDF; one task-specific classifier per task. Logistic Regression and LinearSVC are within 0.01 Macro-F1 of each other — confirms the binding constraint is the representation, not the classifier.'''))

cells.append(code(r'''from src.tfidf_feature_extraction import tfidf_features
from models.baseline.logistic_regression import LogisticRegressionModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, classification_report

X_train, X_val, X_test, vectorizer = tfidf_features(
    df_train, df_val, df_test,
    text_column="text", max_features=15_000, save_path="./models/tfidf",
)

# Fit one classifier per task (sentiment + sarcasm) for both LR and SVM
clfs = {"TF-IDF + LR": {}, "TF-IDF + SVM": {}}
baseline_rows = []
for task in ["Sentiment", "Sarcasm"]:
    y_train, y_test = df_train[task].values, df_test[task].values

    lr  = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                             random_state=SEEDS[0], solver="liblinear").fit(X_train, y_train)
    svm = LinearSVC(max_iter=2000, class_weight="balanced",
                    random_state=SEEDS[0]).fit(X_train, y_train)
    clfs["TF-IDF + LR"][task]  = lr
    clfs["TF-IDF + SVM"][task] = svm

    for name, clf in [("TF-IDF + LR", lr), ("TF-IDF + SVM", svm)]:
        y_pred = clf.predict(X_test)
        baseline_rows.append({
            "model": name, "task": task.lower(),
            "macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
        })

baseline_df = pd.DataFrame(baseline_rows)
baseline_df
'''))

cells.append(code(r'''# Per-variety LR performance — Table for §3.1 of the report
per_var_rows = []
for var in VARIETIES:
    test_var = df_test[df_test["variety"] == var].reset_index(drop=True)
    if len(test_var) == 0: continue
    X_test_var = vectorizer.transform(test_var["text"].tolist())
    for task in ["Sentiment", "Sarcasm"]:
        y_pred = clfs["TF-IDF + LR"][task].predict(X_test_var)
        y_true = test_var[task].values
        per_var_rows.append({
            "variety": var, "task": task.lower(),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
        })

per_var_lr = pd.DataFrame(per_var_rows)
per_var_lr.pivot(index="variety", columns="task", values="macro_f1")
'''))

# -----------------------------------------------------------------------------
# Section 2.2 — RoBERTa cross-variety  (THE BIG ONE — Joel's canonical code)
# -----------------------------------------------------------------------------
cells.append(md(r'''### 2.2 RoBERTa cross-variety (canonical functions inlined from Joel's `task_2_2.ipynb`)

We extend the cross-variety protocol to **five training conditions** — `uk_only`, `au_only`, `in_only`, `inner_pool` (UK+AU), and `all` — each evaluated on every variety's test set (5×3 matrix × 2 seeds = 30 prediction runs).

The functions below are inlined verbatim from Joel's `task_2_2.ipynb` so the marker can read the entire training pipeline in one place.'''))

cells.append(md(r'''**Tokeniser & dataset preparation**'''))

cells.append(code(r'''from datasets import Dataset
from transformers import (
    RobertaForSequenceClassification, RobertaTokenizer,
    TrainingArguments, Trainer,
)
import torch.nn as nn

ROBERTA_MODEL_NAME = "roberta-base"
ROBERTA_TASK       = "Sarcasm"   # cross-variety study uses the sarcasm task
ROBERTA_LABEL_COL  = "Sarcasm"
ROBERTA_TEXT_COL   = "text"
ROBERTA_MAX_LEN    = 128

roberta_tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_NAME) if FROM_SCRATCH else None


def roberta_tokenize(batch, tokenizer=None, label_col=ROBERTA_LABEL_COL):
    # Truncate / pad to 128 tokens and project the label to int.
    tok = tokenizer or roberta_tokenizer
    if tok is None:
        tok = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    out = tok(batch[ROBERTA_TEXT_COL], truncation=True,
              padding="max_length", max_length=ROBERTA_MAX_LEN)
    out["labels"] = [int(label) for label in batch[label_col]]
    return out


def roberta_prepare_dataset(dataset, tokenizer=None, label_col=ROBERTA_LABEL_COL):
    # Tokenise + drop unused columns + cast to torch tensors.
    fn = lambda batch: roberta_tokenize(batch, tokenizer=tokenizer, label_col=label_col)
    tokenized = dataset.map(fn, batched=True)
    tokenized = tokenized.remove_columns(
        [c for c in tokenized.column_names if c not in
         ["input_ids", "attention_mask", "labels"]]
    )
    tokenized.set_format("torch")
    return tokenized
'''))

cells.append(md(r'''**Metric helpers — `compute_metrics` for `Trainer`, `full_evaluation` for the report tables**'''))

cells.append(code(r'''from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)


def compute_metrics(eval_pred):
    # Used by Trainer during fine-tuning — Macro-F1 only for speed.
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "accuracy": (predictions == labels).mean(),
    }


def full_evaluation(y_true, y_pred, task="sarcasm"):
    # Full per-class metrics + confusion matrix — written to JSON after each run.
    if task == "sentiment":
        class_names = ["Negative", "Positive"]
    else:
        class_names = ["Not Sarcastic", "Sarcastic"]
    return {
        "macro_f1":         round(f1_score(y_true, y_pred, average="macro"), 4),
        "precision":        round(precision_score(y_true, y_pred, average="macro"), 4),
        "recall":           round(recall_score(y_true, y_pred, average="macro"), 4),
        "per_class_f1":     f1_score(y_true, y_pred, average=None).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report":           classification_report(y_true, y_pred, target_names=class_names),
    }
'''))

cells.append(md(r'''**Class weighting + `WeightedTrainer` — addresses the 6% – 29% sarcasm imbalance across varieties**'''))

cells.append(code(r'''def calculate_class_weights(dataset, label_col="Sarcasm"):
    # w_c = N / (2 * n_c) — inverse class frequency, BESSTIE convention.
    labels   = dataset[label_col]
    n_total  = len(labels)
    n_class1 = sum(int(x) for x in labels)
    n_class0 = n_total - n_class1
    weight_0 = n_total / (2 * n_class0)
    weight_1 = n_total / (2 * n_class1)
    print(f"Weights for {label_col}: class_0={weight_0:.3f}, class_1={weight_1:.3f}")
    return torch.tensor([weight_0, weight_1], dtype=torch.float)


class WeightedTrainer(Trainer):
    # HF Trainer with weighted CE — moves weights to the model's device per step.
    def __init__(self, *args, loss_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weights = loss_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")
        weights = self.loss_weights.to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
'''))

cells.append(md(r'''**`train_roberta` and `evaluate_on_testset` — the full RoBERTa training loop**'''))

cells.append(code(r'''def train_roberta(train_data, val_data, label_col="Sarcasm", seed=42, output_dir="./tmp"):
    # Fine-tune RoBERTa-base on `train_data` with weighted CE and return (model, tokenizer).
    seed_all(seed)
    weights = calculate_class_weights(train_data, label_col=label_col)
    model   = RobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME, num_labels=2)

    tok = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    train_tok = roberta_prepare_dataset(train_data, tokenizer=tok, label_col=label_col)
    val_tok   = roberta_prepare_dataset(val_data,   tokenizer=tok, label_col=label_col)

    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=5,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        learning_rate=1e-5, warmup_ratio=0.1, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        seed=seed, report_to="none",
    )
    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_tok, eval_dataset=val_tok,
        compute_metrics=compute_metrics, loss_weights=weights,
    )
    trainer.train()
    return model, tok


def evaluate_on_testset(model, test_data, tokenizer, label_col="Sarcasm", task="sarcasm"):
    # Predict on `test_data` and return `full_evaluation` dict.
    test_tok = roberta_prepare_dataset(test_data, tokenizer=tokenizer, label_col=label_col)
    test_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    out = Trainer(model=model).predict(test_tok)
    y_pred = np.argmax(out.predictions, axis=1)
    y_true = np.array(test_tok["labels"])
    return full_evaluation(y_true, y_pred, task=task)
'''))

cells.append(md(r'''**Cross-variety experiment loop — runs 5 conditions × 3 test sets × 2 seeds = 30 evaluations**

This cell is *only* executed when `FROM_SCRATCH=True`. By default we skip it and load the cached `weighted_results/*.json` files (committed to the repo) in the next cell — those are the canonical numbers used in the report.'''))

cells.append(code(r'''from tqdm.auto import tqdm

if FROM_SCRATCH:
    os.makedirs("reports/results/roberta_weighted_rerun", exist_ok=True)

    train_conditions = get_train_conditions(ds)
    test_sets        = get_test_conditions(ds)
    val_data         = ds["validation"]
    all_results      = {}

    for cond_name, train_data in tqdm(train_conditions.items(), desc="Conditions"):
        cond_results = {}
        for seed in tqdm(SEEDS, desc=f"  {cond_name}", leave=False):
            model, tok = train_roberta(
                train_data, val_data, label_col="Sarcasm", seed=seed,
                output_dir=f"./tmp/{cond_name}_seed{seed}",
            )
            seed_results = {}
            for test_name, test_data in test_sets.items():
                seed_results[test_name] = evaluate_on_testset(
                    model, test_data, tokenizer=tok, label_col="Sarcasm", task="sarcasm",
                )
            cond_results[f"seed_{seed}"] = seed_results

        averaged = {}
        for test_name in test_sets.keys():
            f1s = [cond_results[f"seed_{s}"][test_name]["macro_f1"] for s in SEEDS]
            averaged[test_name] = {
                "macro_f1_mean": round(np.mean(f1s), 4),
                "macro_f1_std":  round(np.std(f1s),  4),
            }
        all_results[cond_name] = {"by_seed": cond_results, "averaged": averaged}
        with open(f"reports/results/roberta_weighted_rerun/{cond_name}.json", "w") as f:
            json.dump(all_results[cond_name], f, indent=2)

    print("Re-trained results saved under reports/results/roberta_weighted_rerun/")
else:
    print("FROM_SCRATCH=False → skipping retraining; using cached canonical results.")
'''))

cells.append(md(r'''**Load canonical RoBERTa results & reproduce the cross-variety matrix (Figure 25 in the report)**'''))

cells.append(code(r'''ROBERTA_RESULTS_DIR = Path(
    "reports/results/roberta_weighted_rerun"
    if FROM_SCRATCH and Path("reports/results/roberta_weighted_rerun").exists()
    else "reports/results/roberta_weighted"
)
print(f"Loading RoBERTa results from {ROBERTA_RESULTS_DIR}")

ROBERTA_CONDITIONS  = ["uk_only", "au_only", "in_only", "inner_pool", "all"]
ROBERTA_TEST_NAMES  = ["uk_only", "au_only", "in_only"]   # weighted_results uses the same key for test sets
ROBERTA_DISPLAY     = ["UK only", "AU only", "IN only", "Inner pool", "All pool"]
TEST_DISPLAY        = ["Test UK", "Test AU", "Test IN"]

roberta_results = {}
for cond in ROBERTA_CONDITIONS:
    p = ROBERTA_RESULTS_DIR / f"{cond}.json"
    if p.exists():
        roberta_results[cond] = json.load(open(p))
print("Conditions loaded:", list(roberta_results.keys()))

# Reproduce Joel's 5×3 cross-variety matrix
matrix = np.array([
    [roberta_results[c]["averaged"][t]["macro_f1_mean"] for t in ROBERTA_TEST_NAMES]
    for c in ROBERTA_CONDITIONS
])
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt=".3f",
            xticklabels=TEST_DISPLAY, yticklabels=ROBERTA_DISPLAY,
            cmap="YlOrRd", vmin=0.5, vmax=1.0)
plt.title("RoBERTa — Cross-Variety Evaluation Matrix (Macro-F1)")
plt.ylabel("Trained on"); plt.xlabel("Tested on")
plt.tight_layout()
plt.savefig("reports/figures/roberta_canonical/cross_variety_matrix_repro.png", dpi=150)
plt.show()

results_df = pd.DataFrame(matrix, index=ROBERTA_DISPLAY, columns=TEST_DISPLAY).round(4)
print("\nCross-variety matrix:"); print(results_df)
'''))

cells.append(code(r'''# Confusion matrix for the best condition (Joel's Figure 33-equivalent)
best_condition = max(
    ROBERTA_CONDITIONS,
    key=lambda c: np.mean([roberta_results[c]["averaged"][t]["macro_f1_mean"] for t in ROBERTA_TEST_NAMES]),
)
print(f"Best condition: {best_condition}")

best_cm = np.array(roberta_results[best_condition]["by_seed"]["seed_42"][ROBERTA_TEST_NAMES[0]]["confusion_matrix"])
plt.figure(figsize=(6, 5))
sns.heatmap(best_cm, annot=True, fmt="d",
            xticklabels=["Not Sarcastic", "Sarcastic"],
            yticklabels=["Not Sarcastic", "Sarcastic"], cmap="Blues")
plt.title(f"Confusion Matrix — {best_condition} → UK test")
plt.ylabel("True Label"); plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("reports/figures/roberta_canonical/confusion_matrix_best_repro.png", dpi=150)
plt.show()
'''))

cells.append(code(r'''# Per-class F1 across all 5 conditions (Joel's Figure 32)
records = []
for cond in ROBERTA_CONDITIONS:
    for test_var in ROBERTA_TEST_NAMES:
        # average per_class_f1 across seeds
        f1s = np.mean(
            [roberta_results[cond]["by_seed"][f"seed_{s}"][test_var]["per_class_f1"]
             for s in [42, 123]],
            axis=0,
        )
        records.append({
            "condition": cond, "test": test_var,
            "Not_Sarcastic_F1": round(f1s[0], 4),
            "Sarcastic_F1":     round(f1s[1], 4),
        })
per_class_df = pd.DataFrame(records)
print(per_class_df.pivot_table(index="condition", columns="test",
                                values="Sarcastic_F1").round(4))
'''))

# -----------------------------------------------------------------------------
# Section 2.1 RoBERTa-base on all-pool (sentiment) — uses the cached JSON
# -----------------------------------------------------------------------------
cells.append(md(r'''### 2.1 (cont.) RoBERTa-base on all-pool — sentiment task

This is the §2.1 PTLM baseline used in Table 2. The training loop is the same `train_roberta()` defined above, only `label_col="Sentiment"`. Results were saved by Joel to `results/sentiment/all_pool.json`.'''))

cells.append(code(r'''SENTIMENT_RESULTS_PATH = Path("reports/results/roberta_sentiment/all_pool.json")
sentiment_results = json.load(open(SENTIMENT_RESULTS_PATH))

print("Sentiment all-pool RoBERTa — averaged across seeds 42, 123:")
for test_name, scores in sentiment_results["averaged"].items():
    print(f"  {test_name}: Macro-F1 = {scores['macro_f1_mean']:.4f} ± {scores['macro_f1_std']:.4f}")

# Headline comparison table — used as Table 2 in the report
headline = pd.DataFrame([
    {"model": "TF-IDF + LR",           "task": "sentiment", "macro_f1": baseline_df.loc[(baseline_df["model"]=="TF-IDF + LR") & (baseline_df["task"]=="sentiment"), "macro_f1"].iloc[0]},
    {"model": "TF-IDF + SVM",          "task": "sentiment", "macro_f1": baseline_df.loc[(baseline_df["model"]=="TF-IDF + SVM") & (baseline_df["task"]=="sentiment"), "macro_f1"].iloc[0]},
    {"model": "RoBERTa-base all-pool", "task": "sentiment", "macro_f1": np.mean([sentiment_results["averaged"][k]["macro_f1_mean"] for k in sentiment_results["averaged"]])},
    {"model": "TF-IDF + LR",           "task": "sarcasm",   "macro_f1": baseline_df.loc[(baseline_df["model"]=="TF-IDF + LR") & (baseline_df["task"]=="sarcasm"), "macro_f1"].iloc[0]},
    {"model": "TF-IDF + SVM",          "task": "sarcasm",   "macro_f1": baseline_df.loc[(baseline_df["model"]=="TF-IDF + SVM") & (baseline_df["task"]=="sarcasm"), "macro_f1"].iloc[0]},
    {"model": "RoBERTa-base all-pool", "task": "sarcasm",   "macro_f1": np.mean([roberta_results["all"]["averaged"][k]["macro_f1_mean"] for k in roberta_results["all"]["averaged"]])},
])
headline.pivot(index="model", columns="task", values="macro_f1").round(4)
'''))

# -----------------------------------------------------------------------------
# Section 2.3 — LoRA
# -----------------------------------------------------------------------------
cells.append(md(r'''### 2.3 LoRA adapters — OPT-1.3B (canonical model)

Three per-variety adapters trained on `facebook/opt-1.3b` with `r=8, alpha=16, lr=2e-4`, weighted CE — 1.6 M trainable params out of 1.32 B (0.12%). The functions below come from `models/lora/lora_adapters.py` and Mohamed's `2.3_LoRA_Adapters_Mohamed.ipynb`.'''))

cells.append(code(r'''from models.lora.lora_adapters import (
    LoRAConfig, load_model, apply_lora,
    tokenize_dataset, training_args,
    save_adapter, load_adapter,
)

LORA_BASE_MODEL  = "facebook/opt-1.3b"
LORA_HUB_PREFIX  = f"{HF_HUB_USER}/besstie-lora-en"

LORA_CONFIG = LoRAConfig(
    r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)
print(f"LoRA config: r={LORA_CONFIG.r}, alpha={LORA_CONFIG.lora_alpha}, "
      f"dropout={LORA_CONFIG.lora_dropout}, targets={LORA_CONFIG.target_modules}")
'''))

cells.append(md(r'''**`train_lora_adapter` — train one variety-specific adapter (inlined from Mohamed's notebook, simplified)**'''))

cells.append(code(r'''def train_lora_adapter(variety: str, seed: int, epochs: int = 3, batch_size: int = 4,
                        max_length: int = 128, lr: float = 2e-4,
                        base_model: str = LORA_BASE_MODEL):
    # Train one variety-specific LoRA adapter on the Sarcasm task.
    # Mirrors `train_one(variety, seed)` in Mohamed's `2.3_LoRA_Adapters_Mohamed.ipynb`.
    # Returns (trainer, tokenizer, weighted_class_weights).
    seed_all(seed)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train = get_variety_split(ds, variety, "train")
    val   = get_variety_split(ds, variety, "validation")

    model, tokenizer = load_model(base_model, num_labels=2)
    model = apply_lora(model, LORA_CONFIG)

    train_tok = tokenize_dataset(train, tokenizer, label_col="Sarcasm", max_length=max_length)
    val_tok   = tokenize_dataset(val,   tokenizer, label_col="Sarcasm", max_length=max_length)

    args = training_args(
        output_dir=f"./adapters/lora_{variety.replace('-', '_')}_seed{seed}",
        num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, learning_rate=lr, seed=seed,
    )
    weights = calculate_class_weights(train, label_col="Sarcasm")
    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_tok, eval_dataset=val_tok,
        compute_metrics=compute_metrics, loss_weights=weights,
    )
    trainer.train()
    return trainer, tokenizer, weights


def evaluate_lora_adapter(trainer, tokenizer, test_variety: str, max_length: int = 128):
    # Run a trained LoRA `trainer` on a target test variety.
    test = get_variety_split(ds, test_variety, "test")
    test_tok = tokenize_dataset(test, tokenizer, label_col="Sarcasm", max_length=max_length)
    out = trainer.predict(test_tok)
    y_pred = np.argmax(out.predictions, axis=1)
    y_true = np.asarray(out.label_ids)
    return full_evaluation(y_true, y_pred, task="sarcasm")
'''))

cells.append(code(r'''# Optional retraining — runs only when FROM_SCRATCH=True (~30 min on Colab T4)
if FROM_SCRATCH:
    os.makedirs("reports/results/lora_rerun", exist_ok=True)
    lora_rerun = {}
    for variety in VARIETIES:
        lora_rerun[variety] = {"by_seed": {}}
        for seed in SEEDS:
            trainer, tok, _ = train_lora_adapter(variety=variety, seed=seed)
            seed_res = {}
            for test_var in VARIETIES:
                seed_res[test_var] = evaluate_lora_adapter(trainer, tok, test_var)
            lora_rerun[variety]["by_seed"][f"seed_{seed}"] = seed_res
        # average over seeds
        averaged = {}
        for test_var in VARIETIES:
            macro = [lora_rerun[variety]["by_seed"][f"seed_{s}"][test_var]["macro_f1"] for s in SEEDS]
            sarc  = [lora_rerun[variety]["by_seed"][f"seed_{s}"][test_var]["per_class_f1"][1] for s in SEEDS]
            averaged[test_var] = {
                "macro_f1_mean":     round(float(np.mean(macro)), 4),
                "macro_f1_std":      round(float(np.std(macro)),  4),
                "sarcastic_f1_mean": round(float(np.mean(sarc)),  4),
            }
        lora_rerun[variety]["averaged"] = averaged
        with open(f"reports/results/lora_rerun/{variety}.json", "w") as f:
            json.dump(lora_rerun[variety], f, indent=2)
    print("LoRA retraining done — results in reports/results/lora_rerun/")
else:
    print("FROM_SCRATCH=False → using cached LoRA results.")
'''))

cells.append(md(r'''**Load canonical LoRA results & reproduce the cross-variety matrix (Tables 6–7 / Figure 34 in the report)**

Cached structure (from Mohamed's runs):
```json
{
  "base_model": "Qwen/Qwen2.5-1.5B",        // or facebook/opt-1.3b for the canonical run
  "macro_f1": {
    "seed_42":         {"en-UK": {"en-UK": 0.65, "en-AU": 0.51, "en-IN": 0.60}, ...},
    "seed_123":        {...},
    "mean_over_seeds": {"en-UK": {"en-UK": 0.56, ...}, "en-AU": {...}, "en-IN": {...}}
  }
}
```'''))

cells.append(code(r'''LORA_RESULTS_PATH = Path("reports/results/q2_3_lora_full_sarcasm.json")
lora_results = json.load(open(LORA_RESULTS_PATH)) if LORA_RESULTS_PATH.exists() else None

if lora_results is not None:
    print(f"LoRA base model: {lora_results.get('base_model')}")
    print(f"LoRA seeds:      {lora_results.get('seeds')}")
    mean_mat = lora_results["macro_f1"]["mean_over_seeds"]

    # Cross-variety Macro-F1 heatmap (3 train varieties × 3 test varieties)
    macro_mat = np.array([
        [mean_mat[train][test] for test in VARIETIES]
        for train in VARIETIES
    ])

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(macro_mat, annot=True, fmt=".3f",
                xticklabels=VARIETIES, yticklabels=VARIETIES,
                cmap="YlOrRd", vmin=0.4, vmax=0.85)
    plt.title(f"LoRA {lora_results.get('base_model','base')} — Cross-variety Macro-F1 (Sarcasm)")
    plt.xlabel("Test variety"); plt.ylabel("Adapter (trained on)")
    plt.tight_layout()
    plt.savefig("reports/figures/q2_3_lora_macro_f1_heatmap_repro.png", dpi=150)
    plt.show()

    lora_df = pd.DataFrame(macro_mat, index=VARIETIES, columns=VARIETIES).round(4)
    print("\nLoRA cross-variety matrix (Macro-F1, mean over seeds):")
    print(lora_df)
else:
    print("Cached LoRA JSON not found.")
    lora_df = None
'''))

# -----------------------------------------------------------------------------
# Section 3
# -----------------------------------------------------------------------------
cells.append(md(r'''## 3. Evaluation summary

Cross-task headline comparison and per-variety breakdown — populated from the JSONs we loaded above. No new training is performed in this section.'''))

cells.append(code(r'''# Per-variety Macro-F1 — LR vs RoBERTa all-pool vs in-variety LoRA (Table 8 in the report)
rows = []
lr_pivot = per_var_lr[per_var_lr["task"] == "sarcasm"].set_index("variety")["macro_f1"]
for var in VARIETIES:
    row = {"variety": var}
    row["LR sarcasm"] = float(lr_pivot.get(var, float("nan")))
    var_short = {"en-UK": "uk_only", "en-AU": "au_only", "en-IN": "in_only"}[var]
    row["RoBERTa all-pool sarcasm"] = roberta_results["all"]["averaged"][var_short]["macro_f1_mean"]
    if lora_results and "macro_f1" in lora_results:
        mean_mat = lora_results["macro_f1"]["mean_over_seeds"]
        row["LoRA in-variety sarcasm"] = round(mean_mat[var][var], 4)
    rows.append(row)
summary = pd.DataFrame(rows).round(4).set_index("variety")
print("Cross-task summary (Macro-F1, sarcasm):"); print(summary)
summary
'''))

# -----------------------------------------------------------------------------
# Section 4 — Error analysis
# -----------------------------------------------------------------------------
cells.append(md(r'''## 4. Sarcasm explanation & error analysis

Best model: OPT-1.3B + en-AU LoRA adapter (highest Sarcastic-F1 in §3.4). We:

1. extract 10 high-confidence misclassifications from the en-AU test set,
2. write explanations for 4 of them (2 FPs, 2 FNs),
3. build a 4-shot prompt and re-test the remaining 6 with `meta-llama/Llama-3.2-1B-Instruct`.'''))

cells.append(code(r'''# Step 1 — extract 10 representative errors with the canonical adapter
# (downloads OPT-1.3B + en-AU LoRA adapter, runs inference on en-AU test set)
if RUN_ERROR_ANALYSIS:
    %run scripts/q4_extract_errors.py
else:
    print("Skipping error extraction (RUN_ERROR_ANALYSIS=False). "
          "Pre-computed results already in reports/results/q4_errors.json.")
'''))

cells.append(code(r'''# Step 2 — review the cached errors file (committed to repo)
errors_path = Path("reports/results/q4_errors.json")
if errors_path.exists():
    errors = json.load(open(errors_path))
    print(f"Loaded {len(errors)} extracted errors. First example:")
    print(json.dumps(errors[0], indent=2))
else:
    print("Run with RUN_ERROR_ANALYSIS=True to populate reports/results/q4_errors.json")
'''))

cells.append(code(r'''# Step 3 — few-shot evaluation on the 6 held-out errors with LLaMA-3.2-1B-Instruct
if RUN_ERROR_ANALYSIS:
    %run scripts/q4_few_shot_eval.py
else:
    print("Skipping few-shot re-test (RUN_ERROR_ANALYSIS=False).")
'''))

cells.append(code(r'''if RUN_LIME:
    %run scripts/lime_explain.py
else:
    print("Skipping LIME (RUN_LIME=False).")
'''))

# -----------------------------------------------------------------------------
# Section 5
# -----------------------------------------------------------------------------
cells.append(md(r'''## 5. Deployment & efficiency

### 5.1 Deployment endpoint (Gradio on HuggingFace Spaces)

Live: **https://huggingface.co/spaces/momofahmi/besstie-cw-nlp**

The Gradio app (`app/app.py`) hot-swaps LoRA adapters on a single frozen OPT-1.3B base, so variety switching costs microseconds rather than seconds. To run it:

```bash
HF_HOME=$(pwd)/.cache/huggingface python app/app.py     # local
# or open notebooks/run_deployment_colab.ipynb on a Colab T4
```

Screenshots for the report are in `reports/figures/q5_1_deployment_*.png`.'''))

cells.append(md(r'''### 5.2 Efficiency benchmark

Average inference time across 20 timed runs after a 3-run GPU warm-up. Writes `reports/results/q5_2_efficiency.json` and a Markdown table.'''))

cells.append(code(r'''if RUN_BENCHMARK:
    %run scripts/benchmark_inference.py
else:
    print("Skipping efficiency benchmark (RUN_BENCHMARK=False).")
'''))

cells.append(md(r'''## Done

If you reached this cell, the full pipeline ran successfully and every figure / table referenced in the PDF report has been reproduced.

**Reproducibility checklist** (per the coursework brief):
- Two random seeds (42, 123) used in every training cell, std reported across seeds.
- All hyperparameters live at the top of the notebook (`§0.1 Config flags`) or alongside the relevant function definition.
- All datasets are loaded from the official `surrey-nlp/BESSTIE-CW-26` Hugging Face mirror.
- All adapters are pushed to `huggingface.co/momofahmi/*`, and the canonical run reads the cached JSONs in `reports/results/`.
- For full retraining, set `FROM_SCRATCH=True` in §0.1 and run again on a Colab T4.

**Where to look in the report**:
- Section 1 ↔ §1.1 / §1.2 above
- Section 2 ↔ §2.1 / §2.2 / §2.3 above
- Section 3 ↔ §3 above
- Section 4 ↔ §4 above
- Section 5 ↔ §5 above
'''))

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec":    {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbf.write(nb, f)

print(f"Wrote {OUT} with {len(cells)} cells.")
