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

**Opening from GitHub:** Colab only loads that **one `.ipynb` file** from the URL (the correct branch/path you picked). It does **not** automatically download `src/`, `requirements.txt`, or the rest of the repo. The **first code cell** is what clones the repo (or uses your uploaded zip) so the notebook can import `src` and install dependencies.

**Google “may request access” warning:** Normal for notebooks opened from GitHub; skim the cells you run. These notebooks mainly clone, `pip install`, and load data from Hugging Face.

Notes for Colab:
- Use a GPU runtime: `Runtime -> Change runtime type -> GPU`
- The first cell clones the repo and runs `pip install -r requirements.txt` from the project root.
- Dataset loads from Hugging Face: `load_dataset("surrey-nlp/BESSTIE-CW-26")`
- **Colab notebooks** clone branch **`fiyin/model-pipeline` by default** (so `src/besstie_data_loader.py` exists). Optional env: `GITHUB_REPO`, `REPO_BRANCH`, `REPO_URL`, `REPO_DIR`, `REPO_ZIP`.
- **Zip (no git):** Colab has no folder upload — zip the project on your machine, then **Files** → upload one `.zip` (e.g. `NLP-sequence-classification.zip` under `/content/`, or set `REPO_ZIP`).
- **Private repo:** use a zip, or fork and set `GITHUB_REPO=YourUser/NLP-sequence-classification` for a public fork.
- **Uploading only the `.ipynb` file is not enough** — you need `src/`, `requirements.txt`, and the rest of the repo (or a successful clone).

### Fork to your GitHub for Colab (when `momofahmi/...` is private)

Yes — this is the “fork method”: Colab clones over **anonymous HTTPS**, so the clone URL must point at a repo GitHub serves **without** login (almost always a **public** fork).

1. **Fork** the upstream repo on GitHub: open `https://github.com/momofahmi/NLP-sequence-classification`, click **Fork** (top right), create the fork under your account. You need permission to see the private upstream to fork it (e.g. collaborator).
2. **Branch** `fiyin/model-pipeline` must exist on your fork. If it does not, push it from your laptop:  
   `git push fork fiyin/model-pipeline` (or add your fork as `remote` and push that branch).
3. **Make the fork public** (for testing without tokens): your fork → **Settings → General** (scroll to **Danger zone**) → **Change repository visibility** → **Public**. A **private** fork will hit the same `could not read Username` error on Colab.
4. **Point the notebook at your fork** before the first setup cell runs:
   - **Option A — Colab environment variable:** where Colab lets you set variables, set `GITHUB_REPO` to `YourGitHubUsername/NLP-sequence-classification` (no `https://`, no spaces).
   - **Option B — extra cell above setup:** run once, then run the rest:
     ```python
     import os
     os.environ["GITHUB_REPO"] = "YourGitHubUsername/NLP-sequence-classification"
     ```
5. **Restart** after a failed clone: `Runtime → Restart session`, delete stale folder if needed (`!rm -rf /content/NLP-sequence-classification`), then **Run all** again.

The first notebook cell still runs `pip install -r requirements.txt` after a successful clone into `/content/NLP-sequence-classification`.

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
