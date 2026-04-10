import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import inspect


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.besstie_data_loader import get_variety_split  # noqa: E402


def tokenize_dataset(dataset, tokenizer, label_col: str, max_length: int = 128):
    def _tok(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        out["labels"] = [int(x) for x in batch[label_col]]
        return out

    tokenized = dataset.map(_tok, batched=True)
    keep = {"input_ids", "attention_mask", "labels"}
    tokenized = tokenized.remove_columns([c for c in tokenized.column_names if c not in keep])
    tokenized.set_format("torch")
    return tokenized


def eval_metrics(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    # Quick local sanity check: trains on a tiny subset and evaluates on 3 test varieties.
    # This is not the “final run”; it’s just to verify the pipeline works end-to-end.
    label_col = os.environ.get("LABEL_COL", "Sarcasm")  # Sarcasm or Sentiment
    train_variety = os.environ.get("TRAIN_VARIETY", "en-UK")
    n_train = int(os.environ.get("N_TRAIN", "128"))
    n_val = int(os.environ.get("N_VAL", "64"))
    n_test = int(os.environ.get("N_TEST", "256"))
    epochs = float(os.environ.get("EPOCHS", "1"))
    seed = int(os.environ.get("SEED", "42"))

    np.random.seed(seed)
    torch.manual_seed(seed)

    ds = load_dataset("surrey-nlp/BESSTIE-CW-26")
    train = get_variety_split(ds, train_variety, "train").shuffle(seed=seed).select(range(n_train))
    val = get_variety_split(ds, train_variety, "validation").shuffle(seed=seed).select(range(min(n_val, len(ds["validation"]))))

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    train_tok = tokenize_dataset(train, tokenizer, label_col)
    val_tok = tokenize_dataset(val, tokenizer, label_col)

    args = TrainingArguments(
        output_dir="./tmp/smoke_roberta",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        **({("eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"): "epoch"}),
        save_strategy="no",
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok)
    trainer.train()

    for test_variety in ["en-UK", "en-AU", "en-IN"]:
        test = get_variety_split(ds, test_variety, "test").shuffle(seed=seed).select(range(min(n_test, len(ds["test"]))))
        test_tok = tokenize_dataset(test, tokenizer, label_col)
        pred = trainer.predict(test_tok)
        y_pred = np.argmax(pred.predictions, axis=1)
        y_true = np.asarray(test_tok["labels"])

        m = eval_metrics(y_true, y_pred)
        print(f"\n=== TEST {test_variety} | trained on {train_variety} | label={label_col} ===")
        print({k: m[k] for k in ["macro_f1", "precision_macro", "recall_macro"]})
        print("confusion_matrix:", m["confusion_matrix"])


if __name__ == "__main__":
    main()

