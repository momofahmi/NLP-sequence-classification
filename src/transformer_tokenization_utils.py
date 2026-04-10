"""
Omar - Feature Extraction for Transformer & LoRA
  - Tokenize using RoBERTa tokenizer  → save tokenized datasets
  - Create 30% stratified subsets per variety
  - Tokenize using LoRA LLM tokenizer (Gemma / Phi-2)  → save LoRA subsets
Loads clean_text produced by Yusrah's pipeline.
"""

import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import warnings
warnings.filterwarnings('ignore')


# ------------------------------------------------------------------
# Helper: 30% stratified sample per variety
# ------------------------------------------------------------------
def sample_30_percent_per_variety(df: pd.DataFrame,
                                   variety_col: str = 'variety',
                                   seed: int = 42) -> pd.DataFrame:
    """
    Returns a stratified sample of 30% of rows from each variety,
    preserving class balance within each variety.
    """
    sampled = (
        df.groupby(variety_col, group_keys=False)
        .apply(lambda g: g.sample(frac=0.30, random_state=seed))
    )
    sampled = sampled.reset_index(drop=True)
    print(f"\n30% subset sizes per variety:")
    print(sampled[variety_col].value_counts().to_string())
    return sampled


# ------------------------------------------------------------------
# RoBERTa tokenization
# ------------------------------------------------------------------
def tokenize_roberta(df_train: pd.DataFrame,
                      df_val: pd.DataFrame,
                      df_test: pd.DataFrame,
                      text_col: str = 'clean_text',
                      label_cols: list = None,
                      max_length: int = 128,
                      save_path: str = './tokenized/roberta') -> DatasetDict:
    """
    Tokenize all splits using roberta-base.
    Saves tokenized DatasetDict to disk.

    Args:
        df_train / df_val / df_test : DataFrames from Yusrah's pipeline
        text_col   : column with cleaned text (Yusrah produces 'clean_text')
        label_cols : list of label columns to keep, e.g. ['Sentiment', 'Sarcasm']
        max_length : max token length for truncation/padding
        save_path  : where to write the tokenized dataset

    Returns:
        DatasetDict with keys 'train', 'validation', 'test'
    """
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

    # Rename label columns to match HuggingFace Trainer conventions
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


# ------------------------------------------------------------------
# LoRA subset tokenization
# ------------------------------------------------------------------
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
    """
    Create 30% stratified subsets per variety, then tokenize using
    the specified LLM tokenizer (Gemma or Phi-2).

    Args:
        model_name : 'google/gemma-2b' or 'microsoft/phi-2'
        sample_frac: fraction to sample per variety (default 0.30)
        save_path  : output directory
    """
    from transformers import AutoTokenizer

    if label_cols is None:
        label_cols = ['Sentiment', 'Sarcasm']

    print(f"\nLoading tokenizer for LoRA: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Some LLMs don't set a pad token — use eos_token as fallback
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  ⚠️  pad_token not set — using eos_token: '{tokenizer.eos_token}'")

    # Sample 30% per variety from train; keep val/test as is (they are already small)
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
