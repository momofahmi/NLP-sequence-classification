# NLP-sequence-classification
**BESSTIE: Sentiment & Sarcasm Classification across English Varieties**  
University of Surrey, Semester 2, 2026

## Colab
Open in Colab (branch `fiyin/model-pipeline`):

- [1.1 EDA](https://colab.research.google.com/github/momofahmi/NLP-sequence-classification/blob/fiyin/model-pipeline/notebooks/1.1_EDA_Distributions_Yusrah_Omar.ipynb)
- [2.1 Baseline TF-IDF](https://colab.research.google.com/github/momofahmi/NLP-sequence-classification/blob/fiyin/model-pipeline/notebooks/2.1_Baseline_TFIDF_LogReg_Yusrah_Omar.ipynb)
- [2.2 RoBERTa cross-variety](https://colab.research.google.com/github/momofahmi/NLP-sequence-classification/blob/fiyin/model-pipeline/notebooks/2.2_RoBERTa_CrossVariety_Joel_Fiyin.ipynb)
- [2.3 LoRA](https://colab.research.google.com/github/momofahmi/NLP-sequence-classification/blob/fiyin/model-pipeline/notebooks/2.3_LoRA_Adapters_Mohamed.ipynb)

Use a **GPU** runtime. The first code cell in the training notebooks clones the repo and runs `pip install -r requirements.txt`. The BESSTIE dataset loads from Hugging Face: `surrey-nlp/BESSTIE-CW-26`.  
For a **private** GitHub repo, add a Colab secret **`GITHUB_TOKEN`** (classic PAT with `repo` scope) and enable notebook access for it, or upload a project zip to `/content/`. Optional env: **`GITHUB_REPO`** (`owner/repo`) and **`REPO_BRANCH`**.

## Local setup
```bash
git clone https://github.com/momofahmi/NLP-sequence-classification.git
cd NLP-sequence-classification
git checkout fiyin/model-pipeline   # if you need this branch
pip install -r requirements.txt
```

```python
from datasets import load_dataset
ds = load_dataset("surrey-nlp/BESSTIE-CW-26")
```
