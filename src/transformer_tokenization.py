import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import warnings
warnings.filterwarnings('ignore')


def sample_30_percent_per_variety(df: pd.DataFrame,
                                   variety_col: str = 'variety',
                                   seed: int = 42) -> pd.DataFrame:
                                     
    sampled = (
        df.groupby(variety_col, group_keys=False)
        .apply(lambda g: g.sample(frac=0.30, random_state=seed))
    )
    sampled = sampled.reset_index(drop=True)
    print(f"\n30% subset sizes per variety:")
    print(sampled[variety_col].value_counts().to_string())
    return sampled

def tokenize_roberta(df_train: pd.DataFrame,
                      df_val: pd.DataFrame,
                      df_test: pd.DataFrame,
                      text_col: str = 'clean_text',
                      label_cols: list = None,
                      max_length: int = 128,
                      save_path: str = './tokenized/roberta') -> DatasetDict:
  
    from transformers import AutoTokenizer

    if label_cols is None:
        label_cols = ['Sentiment', 'Sarcasm']

    model_name = 'roberta-base'
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _to_hf_dataset(df: pd.DataFrame) -> Dataset:
        keep_cols = [text_col] + [c for c in label_cols if c in df.columns]
        if 'variety' in df.columns:
            keep_cols.append('variety')
        return Dataset.from_pandas(df[keep_cols].reset_index(drop=True))

    def _tokenize_fn(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding='max_length',
            max_length=max_length,
        )

    dataset = DatasetDict({
        'train':      _to_hf_dataset(df_train),
        'validation': _to_hf_dataset(df_val),
        'test':       _to_hf_dataset(df_test),
    })

    print(f"Tokenizing with max_length={max_length}...")
    tokenized = dataset.map(
        _tokenize_fn,
        batched=True,
        desc='Tokenizing',
    )

    for col in label_cols:
        if col in tokenized['train'].column_names:
            tokenized = tokenized.rename_column(col, col.lower())

    os.makedirs(save_path, exist_ok=True)
    tokenized.save_to_disk(save_path)
    print(f"✅ RoBERTa tokenized dataset saved to: {save_path}")
    print(f"   Train: {len(tokenized['train']):,}  "
          f"Val: {len(tokenized['validation']):,}  "
          f"Test: {len(tokenized['test']):,}")

    return tokenized

def tokenize_lora_subset(df_train: pd.DataFrame,
                          df_val: pd.DataFrame,
                          df_test: pd.DataFrame,
                          text_col: str = 'clean_text',
                          label_cols: list = None,
                          model_name: str = 'google/gemma-2b',
                          max_length: int = 256,
                          sample_frac: float = 0.30,
                          save_path: str = './tokenized/lora',
                          seed: int = 42) -> DatasetDict:
    
    from transformers import AutoTokenizer

    if label_cols is None:
        label_cols = ['Sentiment', 'Sarcasm']

    print(f"\nLoading tokenizer for LoRA: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  ⚠️  pad_token not set — using eos_token: '{tokenizer.eos_token}'")

    
    train_subset = sample_30_percent_per_variety(
        df_train, variety_col='variety', seed=seed
    )
    val_subset = sample_30_percent_per_variety(
        df_val, variety_col='variety', seed=seed
    )
    test_subset = sample_30_percent_per_variety(
        df_test, variety_col='variety', seed=seed
    )

    def _to_hf_dataset(df: pd.DataFrame) -> Dataset:
        keep_cols = [text_col] + [c for c in label_cols if c in df.columns]
        if 'variety' in df.columns:
            keep_cols.append('variety')
        return Dataset.from_pandas(df[keep_cols].reset_index(drop=True))

    def _tokenize_fn(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding='max_length',
            max_length=max_length,
        )

    dataset = DatasetDict({
        'train':      _to_hf_dataset(train_subset),
        'validation': _to_hf_dataset(val_subset),
        'test':       _to_hf_dataset(test_subset),
    })

    print(f"Tokenizing LoRA subsets with max_length={max_length}...")
    tokenized = dataset.map(
        _tokenize_fn,
        batched=True,
        desc='Tokenizing LoRA subset',
    )

    model_tag = model_name.replace('/', '_')
    out_path = os.path.join(save_path, model_tag)
    os.makedirs(out_path, exist_ok=True)
    tokenized.save_to_disk(out_path)
    print(f"✅ LoRA tokenized subset saved to: {out_path}")
    print(f"   Train: {len(tokenized['train']):,}  "
          f"Val: {len(tokenized['validation']):,}  "
          f"Test: {len(tokenized['test']):,}")

    return tokenized
