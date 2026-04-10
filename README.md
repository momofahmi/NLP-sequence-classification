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
- The first cell clones the repo and runs `pip install -r requirements.txt` from the project root.
- Dataset loads from Hugging Face: `load_dataset("surrey-nlp/BESSTIE-CW-26")`
- **Private GitHub repo:** Colab cannot clone without credentials.
  - In Colab: sidebar **key icon → Secrets** → add a secret whose **name is exactly** `GITHUB_TOKEN` (the label you gave the token on GitHub does not matter). Paste the `ghp_…` value. **Turn on “Notebook access”** for that secret, or the code cannot read it.
  - Your GitHub account must **already have read access** to that private repo (collaborator). A token only proves *you*; it does not unlock someone else’s private repo. If needed, fork the repo to your account and set environment variable `GITHUB_REPO=YourUser/NLP-sequence-classification` (and optionally `REPO_BRANCH=fiyin/model-pipeline`).
  - **Zip:** Colab has no “upload folder” button. On your laptop zip the project into **one file**, then in Colab open the **Files** tab (folder icon) → **upload** → choose that `.zip`. Name it `NLP-sequence-classification.zip` or set `REPO_ZIP` to its path under `/content/`.
- **Uploading only the `.ipynb` file is not enough** — you need `src/`, `requirements.txt`, and the rest of the repo (or a successful clone).

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
