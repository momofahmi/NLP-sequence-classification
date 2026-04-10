import os
import torch
from dataclasses import dataclass
from typing import Optional
import inspect

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback,
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
    # Open-weight, generally ungated option in the 1–3B range
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    # Small model for fast local smoke tests (not for final Q2.3 results)
    "opt-125m": "facebook/opt-125m",
}

VARIETIES = ["en-UK", "en-AU", "en-IN"]

HF_USERNAME = "momofahmi"


# LoRA configuration
@dataclass
class LoRAConfig:
    r: int = 8 

    lora_alpha: int = 16

    lora_dropout: float = 0.1

    target_modules: list = None

    task_type: TaskType = TaskType.SEQ_CLS

    # false during training to make mistake
    inference_mode: bool = False

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]
    
    # convert to PEFT 
    def to_peft_config(self) -> LoraConfig:
        return LoraConfig(
            r               = self.r,
            lora_alpha      = self.lora_alpha,
            lora_dropout    = self.lora_dropout,
            target_modules  = self.target_modules,
            task_type       = self.task_type,
            inference_mode  = self.inference_mode,
            bias            = "none", 
        )


# model loading
def load_model(
    model_key : str,
    num_labels: int = 2, # label possible
    device_map: str = "auto",
) -> tuple:
   
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model key '{model_key}'")

    model_id = SUPPORTED_MODELS[model_key]
    print(f"[load_model] Loading {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # decoder models don't have a pad token by default so eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[load_model] Set pad_token = eos_token (required for decoder models)")

    # load the model 
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels = num_labels,
        device_map = device_map,
        # transformers>=5 prefers `dtype` over `torch_dtype`
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[load_model] Loaded {model_id}")
    print(f"[load_model] Total parameters: {total_params / 1e9:.2f}B")

    return model, tokenizer

# freeze base model and apply LoRA adapters
def apply_lora(model, lora_config: LoRAConfig = None) -> object:
  
    if lora_config is None:
        lora_config = LoRAConfig()

    peft_config = lora_config.to_peft_config()
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    return model


# tokenisation 
def tokenize_dataset(dataset, tokenizer, label_col: str = "Sarcasm", max_length: int = 128):

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation = True,
            padding    = "max_length",
            max_length = max_length,
        )
        # create labels column
        tokenized["labels"] = [int(l) for l in examples[label_col]]
        return tokenized

    # remove original columns (keep only what the model needs)
    cols_to_remove = [c for c in dataset.column_names if c != label_col]
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
    tokenized = tokenized.remove_columns([label_col]) # remove original label column after creating labels
    tokenized.set_format("torch")

    return tokenized


# training arguments
def training_args(
    output_dir : str,
    variety    : str,
    seed       : int = 42,
    epochs     : int = 3,
    batch_size : int = 8,
    lr         : float = 2e-4,
) -> TrainingArguments:

    # transformers 5 uses `eval_strategy`; older versions use `evaluation_strategy`
    sig = inspect.signature(TrainingArguments.__init__)
    eval_key = "eval_strategy" if "eval_strategy" in sig.parameters else "evaluation_strategy"

    kwargs = dict(
        output_dir=output_dir,  # lora adapters will be saved here
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        seed=seed,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        run_name=f"lora-{variety}-seed{seed}",
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        remove_unused_columns=False,
    )
    kwargs[eval_key] = "epoch"

    return TrainingArguments(**kwargs)


# adapter saving and loading
def save_adapter(model, variety: str, output_dir: str = "./adapters"):
    
    save_path = os.path.join(output_dir, variety.replace("-", "_"))
    model.save_pretrained(save_path)
    print(f"[save_adapter] Adapter saved to {save_path}")
    print(f"[save_adapter] Size: {get_size(save_path):.1f} MB")



def load_adapter(base_model, variety: str, adapter_dir: str = "./adapters"):
   
    adapter_path = os.path.join(adapter_dir, variety.replace("-", "_"))
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print(f"[load_adapter] Loaded adapter for {variety} from {adapter_path}")
    return model


def push_adapter_to_hub(model, variety: str, hf_username: str = HF_USERNAME):
    
    repo_name = f"besstie-lora-{variety.lower()}"
    hub_path  = f"{hf_username}/{repo_name}"

    print(f"[push_adapter_to_hub] Pushing to {hub_path}...")
    model.push_to_hub(hub_path)
    print(f"[push_adapter_to_hub] Done — load with:")
    print(f"  PeftModel.from_pretrained(base_model, '{hub_path}')")


#return size of directory in MB
def get_size(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)



if __name__ == "__main__":
    import sys
    sys.path.append("..")  

    from datasets import load_dataset
    from src.training_utils import class_weights, new_weighted_class

    from dotenv import load_dotenv
    from huggingface_hub import login

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        login()

    MODEL_KEY  = "llama-3b"  
    VARIETY    = "en-UK"
    TASK       = "Sarcasm"
    MAX_LENGTH = 128
    SEED       = 42

    print("="*55)
    print("LoRA adapter — quick training test")
    print(f"Model   : {MODEL_KEY}")
    print(f"Variety : {VARIETY}")
    print(f"Task    : {TASK}")
    print("="*55)

    # load data
    print("\n[1/5] Loading dataset...")
    ds = load_dataset("surrey-nlp/BESSTIE-CW-26")
    train = ds["train"].filter(lambda x: x["variety"] == VARIETY)
    val   = ds["validation"].filter(lambda x: x["variety"] == VARIETY)

    print(f"   train: {len(train)} examples (reduced for test)")
    print(f"   val  : {len(val)} examples")

    # load model
    print("\n[2/5] Loading base model...")
    model, tokenizer = load_model(MODEL_KEY, num_labels=2)

    # apply lora 
    print("\n[3/5] Applying LoRA...")
    model = apply_lora(model, LoRAConfig())

    # tokenize 
    print("\n[4/5] Tokenizing...")
    train_tok = tokenize_dataset(train, tokenizer, TASK, MAX_LENGTH)
    val_tok   = tokenize_dataset(val,   tokenizer, TASK, MAX_LENGTH)

    # train 
    print("\n[5/5] Training (1 epoch)...")
    weights         = class_weights(train, TASK)
    WeightedTrainer = new_weighted_class(weights)

    args = training_args(
        output_dir = "./test_adapter",
        variety    = VARIETY,
        seed       = SEED,
        epochs     = 3,      
        batch_size = 8,     
        lr         = 2e-4,
    )

    trainer = WeightedTrainer(
        model         = model,
        args          = args,
        train_dataset = train_tok,
        eval_dataset  = val_tok,
    )

    trainer.train()

    print("\nTest complete.")
    print("If you got here without errors the full pipeline works.")
    print("Adapter saved to ./test_adapter/")