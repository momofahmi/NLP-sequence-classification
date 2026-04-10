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
- **Private repo:** Colab cannot `git clone` it without credentials. Use a **zip** (see below), or a **public** GitHub repo that contains the same code (see “Private fork” section).
- **Uploading only the `.ipynb` file is not enough** — you need `src/`, `requirements.txt`, and the rest of the repo (or a successful clone).

### Colab needs a **public** clone URL (or a zip)

`git clone https://github.com/...` in Colab uses **no** GitHub password. So the repo must be **public**, or clone fails with `could not read Username`.

### If your fork is private (common when upstream is private)

GitHub often **does not let you change a fork to public** when the parent repository is private. Your private fork (e.g. `TheFinix13/NLP-coursework`) **still cannot be cloned from Colab** until something is public.

Use one of these:

**A — Zip (simplest)**  
On your laptop, zip the project folder (include `src/`, `requirements.txt`, etc.), upload `NLP-sequence-classification.zip` to Colab’s **Files** (`/content/`), then run the first notebook cell.

**B — New public repo (not a fork)**  
Create a **brand-new** repository on GitHub, choose **Public**, any name (e.g. `besstie-colab-mirror`). Do **not** use “Fork”; it should not be linked as a fork of the private group repo. From your machine (where you already have the code):

```bash
cd /path/to/your/local/clone
git remote add colab_public https://github.com/TheFinix13/YOUR_NEW_PUBLIC_REPO.git
git push colab_public fiyin/model-pipeline:fiyin/model-pipeline
```

Then in Colab, before the setup cell:

```python
import os
os.environ["GITHUB_REPO"] = "TheFinix13/YOUR_NEW_PUBLIC_REPO"
os.environ["REPO_BRANCH"] = "fiyin/model-pipeline"
```

(If you pushed that branch as `main` on the public repo, set `REPO_BRANCH` to `main` instead.)

**C — Ask upstream**  
Ask the owner to make the group repo **public** for marking, or add a **public** read-only mirror the course can clone.

### If you do have a **public** fork or mirror

Set `GITHUB_REPO` to the **`owner/repo`** slug (repo name can differ from upstream, e.g. `TheFinix13/NLP-coursework`). Run this **before** the first setup cell, or use Colab environment variables:

```python
import os
os.environ["GITHUB_REPO"] = "TheFinix13/NLP-coursework"  # example; repo must be public
```

Then **Runtime → Restart session**, `!rm -rf /content/NLP-sequence-classification` if a bad clone exists, and **Run all** again.

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
