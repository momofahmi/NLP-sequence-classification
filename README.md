# NLP-sequence-classification
**BESSTIE: Sentiment & Sarcasm Classification across English Varieties**
University of Surrey ,Semester 2, 2026

## Run in Colab (recommended)
Open notebooks directly in Google Colab (fastest for group members).

- `notebooks/1.1_EDA_Distributions_Yusrah_Omar.ipynb`: Dataset EDA (Q1.1)
  - Open in Colab: `https://colab.research.google.com/github/momofahmi/NLP-sequence-classification/blob/fiyin/model-pipeline/notebooks/1.1_EDA_Distributions_Yusrah_Omar.ipynb`
- `notebooks/2.1_Baseline_TFIDF_LogReg_Yusrah_Omar.ipynb`: Classical baseline (Q2.1)
  - Open in Colab: `https://colab.research.google.com/github/momofahmi/NLP-sequence-classification/blob/fiyin/model-pipeline/notebooks/2.1_Baseline_TFIDF_LogReg_Yusrah_Omar.ipynb`
- `notebooks/2.2_RoBERTa_CrossVariety_Joel_Fiyin.ipynb`: RoBERTa cross-variety (Q2.1/Q2.2)
  - Open in Colab: `https://colab.research.google.com/github/momofahmi/NLP-sequence-classification/blob/fiyin/model-pipeline/notebooks/2.2_RoBERTa_CrossVariety_Joel_Fiyin.ipynb`
- `notebooks/2.3_LoRA_Adapters_Mohamed.ipynb`: LoRA adapters (Q2.3)
  - Open in Colab: `https://colab.research.google.com/github/momofahmi/NLP-sequence-classification/blob/fiyin/model-pipeline/notebooks/2.3_LoRA_Adapters_Mohamed.ipynb`

Colab links target branch **`fiyin/model-pipeline`** (merge to `main` later and switch URLs if needed).

Notes for Colab:
- Use a GPU runtime: `Runtime -> Change runtime type -> GPU`
- First cell should install dependencies: `!pip install -r requirements.txt`
- Dataset loads from Hugging Face: `load_dataset("surrey-nlp/BESSTIE-CW-26")`

## Setup
```bash
git clone https://github.com/momofahmi/NLP-sequence-classification.git
cd NLP-SEQUENCE-CLASSIFICATION
pip install -r requirements.txt
```

How to get the dataset in your local folder:
```python
from datasets import load_dataset
ds = load_dataset("surrey-nlp/BESSTIE-CW-26")
```
