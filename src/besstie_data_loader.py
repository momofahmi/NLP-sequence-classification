from datasets import load_dataset
import pandas as pd

#Mohammad
DATASET_ID = "surrey-nlp/BESSTIE-CW-26"
VARIETIES  = ["en-AU", "en-IN", "en-UK"]


def load_besstie():
    return load_dataset(DATASET_ID)

def get_variety_split(ds, variety: str, split: str):
    return ds[split].filter(lambda x: x["variety"] == variety)

# ---------------------------------------------------------------------
# Helpers for experiments (cross-variety / pooled training conditions)
# ---------------------------------------------------------------------
def get_all_varieties(ds, split: str):
    """Return the requested split for all varieties."""
    return ds[split]

def get_inner_circle_pool(ds, split: str):
    """Pool en-UK and en-AU samples for the requested split."""
    return ds[split].filter(lambda x: x["variety"] in ["en-UK", "en-AU"])

def get_train_conditions(ds):
    """Return training splits used for cross-variety experiments."""
    return {
        "uk_only": get_variety_split(ds, "en-UK", "train"),
        "au_only": get_variety_split(ds, "en-AU", "train"),
        "in_only": get_variety_split(ds, "en-IN", "train"),
        "inner_pool": get_inner_circle_pool(ds, "train"),
        "all": get_all_varieties(ds, "train"),
    }

def get_test_conditions(ds):
    """Return test splits used for cross-variety experiments."""
    return {
        "uk_test": get_variety_split(ds, "en-UK", "test"),
        "au_test": get_variety_split(ds, "en-AU", "test"),
        "in_test": get_variety_split(ds, "en-IN", "test"),
    }

#Yusrah - 30/03/2026
def get_BESSTIE_splits():
  #ds = load_dataset("surrey-nlp/BESSTIE-CW-26")
  ds = load_besstie() #using your function defined above

  #converting the splits to pandas dataframe
  df_train = ds["train"].to_pandas()
  df_val = ds["validation"].to_pandas()
  df_test = ds["test"].to_pandas()

  #adding a split column to preserve which rows came from which split
  df_train["split"] = "train"
  df_val["split"] = "validation"
  df_test["split"] = "test"
  #concatenating all splits into a full dataset for analysis/visualization purposes
  df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

  return df_all, df_train, df_val,df_test

#To use this function, simply call
#df_all, df_train, df_val,df_test = get_BESSTIE_splits()


