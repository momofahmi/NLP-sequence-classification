from datasets import load_dataset
import pandas as pd

#Mohammad
DATASET_ID = "surrey-nlp/BESSTIE-CW-26"
VARIETIES  = ["en-AU", "en-IN", "en-UK"]


def load_besstie():
    return load_dataset(DATASET_ID)

def get_variety_split(ds, variety: str, split: str):
    return ds[split].filter(lambda x: x["variety"] == variety)

#Yusrah - 30/03/2026
def get_BESSTIE_splits():
  ds = load_besstie() 
  df_train = ds["train"].to_pandas()
  df_val = ds["validation"].to_pandas()
  df_test = ds["test"].to_pandas()

  df_train["split"] = "train"
  df_val["split"] = "validation"
  df_test["split"] = "test"

  df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

  return df_all, df_train, df_val,df_test
