import os
import sys
from dataclasses import asdict

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.besstie_data_loader import get_variety_split  # noqa: E402
from src.training_utils import class_weights, new_weighted_class  # noqa: E402
from models.lora.lora_adapters import (  # noqa: E402
    LoRAConfig,
    apply_lora,
    load_model,
    tokenize_dataset,
    training_args,
)


def eval_metrics(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    # Quick sanity check to ensure LoRA training loop starts and produces logits.
    # For speed, default base model is tiny (opt-125m). For Q2.3, switch to qwen2.5-1.5b.
    model_key = os.environ.get("MODEL_KEY", "opt-125m")
    variety = os.environ.get("VARIETY", "en-UK")
    label_col = os.environ.get("LABEL_COL", "Sarcasm")
    seed = int(os.environ.get("SEED", "42"))
    n_train = int(os.environ.get("N_TRAIN", "64"))
    n_val = int(os.environ.get("N_VAL", "64"))
    n_test = int(os.environ.get("N_TEST", "128"))
    epochs = float(os.environ.get("EPOCHS", "1"))

    np.random.seed(seed)
    torch.manual_seed(seed)

    ds = load_dataset("surrey-nlp/BESSTIE-CW-26")
    train = get_variety_split(ds, variety, "train").shuffle(seed=seed).select(range(n_train))
    val = get_variety_split(ds, variety, "validation").shuffle(seed=seed).select(range(min(n_val, len(ds["validation"]))))
    test = get_variety_split(ds, variety, "test").shuffle(seed=seed).select(range(min(n_test, len(ds["test"]))))

    base_model, tokenizer = load_model(model_key, num_labels=2)
    lora_cfg = LoRAConfig()
    model = apply_lora(base_model, lora_cfg)

    train_tok = tokenize_dataset(train, tokenizer, label_col=label_col, max_length=128)
    val_tok = tokenize_dataset(val, tokenizer, label_col=label_col, max_length=128)
    test_tok = tokenize_dataset(test, tokenizer, label_col=label_col, max_length=128)

    weights = class_weights(train, label_col)
    WeightedTrainer = new_weighted_class(weights)

    args = training_args(
        output_dir="./tmp/smoke_lora",
        variety=variety,
        seed=seed,
        epochs=epochs,
        batch_size=4,
        lr=2e-4,
    )
    args.save_strategy = "no"
    args.logging_steps = 5
    args.report_to = "none"

    trainer = WeightedTrainer(model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok)
    trainer.train()

    pred = trainer.predict(test_tok)
    y_pred = np.argmax(pred.predictions, axis=1)
    y_true = np.asarray(test_tok["labels"])
    m = eval_metrics(y_true, y_pred)

    print("\n=== LoRA smoke test ===")
    print("base:", model_key)
    print("variety:", variety)
    print("label:", label_col)
    print("lora_config:", asdict(lora_cfg))
    print({k: m[k] for k in ["macro_f1", "precision_macro", "recall_macro"]})
    print("confusion_matrix:", m["confusion_matrix"])


if __name__ == "__main__":
    main()

