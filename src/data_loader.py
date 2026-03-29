from datasets import load_dataset

DATASET_ID = "surrey-nlp/BESSTIE-CW-26"
VARIETIES  = ["en-AU", "en-IN", "en-UK"]


def load_besstie():
    return load_dataset(DATASET_ID)


def get_variety_split(ds, variety: str, split: str):
    return ds[split].filter(lambda x: x["variety"] == variety)