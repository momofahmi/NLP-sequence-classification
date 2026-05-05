# Mohamed Fahmi Ahmed

import os
import sys
import torch
from dataclasses import dataclass

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)

SUPPORTED_MODELS = {
    "llama-1b" : "meta-llama/Llama-3.2-1B",   
    "llama-3b" : "meta-llama/Llama-3.2-3B",  
    "opt-1.3b" : "facebook/opt-1.3b",         
    "opt-125m" : "facebook/opt-125m",         
}

VARIETIES   = ["en-UK", "en-AU", "en-IN"]
HF_USERNAME = "momofahmi"


# LoRA Configuration
@dataclass
class LoRAConfig:
    r              : int   = 8
    lora_alpha     : int   = 16
    lora_dropout   : float = 0.1
    target_modules : list  = None
    task_type      : TaskType = TaskType.SEQ_CLS
    inference_mode : bool  = False

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]

    def to_peft_config(self) -> LoraConfig:
        # convert to HuggingFace PEFT LoraConfig object
        return LoraConfig(
            r              = self.r,
            lora_alpha     = self.lora_alpha,
            lora_dropout   = self.lora_dropout,
            target_modules = self.target_modules,
            task_type      = self.task_type,
            inference_mode = self.inference_mode,
            bias           = "none",
        )


# model loading 
def load_model(
    model_key  : str,
    num_labels : int = 2,
    device_map : str = "auto",
) -> tuple:
  
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_key}'. "
            f"Choose from: {list(SUPPORTED_MODELS.keys())}"
        )

    model_id = SUPPORTED_MODELS[model_key]
    print(f"[load_model] Loading {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # add pad token to decoder models 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[load_model] Set pad_token = eos_token (decoder model fix)")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels = num_labels,
        device_map = device_map,
        dtype      = torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    total = sum(p.numel() for p in model.parameters())
    print(f"[load_model] Loaded {model_id}")
    print(f"[load_model] Total parameters: {total/1e9:.2f}B")

    return model, tokenizer


# apply LoRA 
def apply_lora(model, lora_config: LoRAConfig = None) -> object:
    
    if lora_config is None:
        lora_config = LoRAConfig()

    model = get_peft_model(model, lora_config.to_peft_config())

    # cast adapter params to float32 for stable gradient computation
    # base model stays float16 (saves VRAM), only trainable A and B go to float32
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    model.print_trainable_parameters()
    return model


# tokenisation 
def tokenize_dataset(
    dataset,
    tokenizer,
    label_col  : str = "Sarcasm",
    max_length : int = 128,
):
    
    def tokenize_fn(examples):
        out = tokenizer(
            examples["text"],
            truncation = True,
            padding    = "max_length",
            max_length = max_length,
        )
        # huggingFace trainer looks for 'labels' column specifically
        out["labels"] = [int(l) for l in examples[label_col]]
        return out

    cols_to_remove = [c for c in dataset.column_names if c != label_col]
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
    tokenized = tokenized.remove_columns([label_col])
    tokenized.set_format("torch")
    return tokenized


# training arguments 
def training_args(
    output_dir : str,
    variety    : str,
    seed       : int   = 42,
    epochs     : int   = 3,
    batch_size : int   = 8,
    lr         : float = 2e-4,
) -> TrainingArguments:
   
    return TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size * 2,
        learning_rate               = lr,
        seed                        = seed,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        logging_steps               = 10,
        run_name                    = f"lora-{variety}-seed{seed}",
        fp16                        = torch.cuda.is_available(),
        push_to_hub                 = False,
        remove_unused_columns       = False,
        save_total_limit            = 1,
        report_to                   = "none",
    )


# adapter 
def save_adapter(model, variety: str, output_dir: str = "./adapters"):
    path = os.path.join(output_dir, variety.replace("-", "_"))
    model.save_pretrained(path)
    print(f"[save_adapter] Saved to {path} ({get_size(path):.1f} MB)")


def load_adapter(base_model, adapter_path: str):
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print(f"[load_adapter] Loaded from {adapter_path}")
    return model


def push_adapter_to_hub(model, tokenizer, variety: str, model_key: str):
    repo = f"{HF_USERNAME}/besstie-lora-{variety.lower()}-{model_key}"
    print(f"[push_adapter_to_hub] Pushing to {repo}...")
    model.push_to_hub(repo)
    tokenizer.push_to_hub(repo)
    print(f"[push_adapter_to_hub] Done — load with:")
    print(f"  PeftModel.from_pretrained(base_model, '{repo}')")


def get_size(path: str) -> float:
    #return directory size in MB
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)


