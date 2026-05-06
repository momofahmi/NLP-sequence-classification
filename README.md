# NLP-sequence-classification
**BESSTIE: Sentiment & Sarcasm Classification across English Varieties**  
University of Surrey, Semester 2, 2026

**Coursework checklist (tick in-repo):** [docs/coursework_checklist.md](docs/coursework_checklist.md)  
**Report outline:** [docs/report_outline.md](docs/report_outline.md) · **Trim guide:** [docs/REPORT_TRIM.md](docs/REPORT_TRIM.md) · **Main notebook plan:** [docs/MAIN_NOTEBOOK_PLAN.md](docs/MAIN_NOTEBOOK_PLAN.md)

## Public mirror

This branch lives on a public mirror so anyone (incl. the marker) can clone without auth:
**https://github.com/TheFinix13/NLP-coursework** (default branch: `main`).

The original team repo `momofahmi/NLP-sequence-classification` is private and may not be reachable from Colab without a `GITHUB_TOKEN`. The two repositories share content but the public one is the canonical source for running the notebooks end-to-end.

## Colab

Open the canonical entry point on a **T4 GPU** runtime:

- **Main pipeline (run-everything):** [notebooks/main.ipynb](https://colab.research.google.com/github/TheFinix13/NLP-coursework/blob/main/notebooks/main.ipynb)
- 1.1 EDA: [notebooks/1.1_EDA_Distributions_Yusrah_Omar.ipynb](https://colab.research.google.com/github/TheFinix13/NLP-coursework/blob/main/notebooks/1.1_EDA_Distributions_Yusrah_Omar.ipynb)
- 2.1 Baseline TF-IDF: [notebooks/2.1_Baseline_TFIDF_LogReg_Yusrah_Omar.ipynb](https://colab.research.google.com/github/TheFinix13/NLP-coursework/blob/main/notebooks/2.1_Baseline_TFIDF_LogReg_Yusrah_Omar.ipynb)
- 2.2 RoBERTa cross-variety: [notebooks/2.2_RoBERTa_CrossVariety_Joel_Fiyin.ipynb](https://colab.research.google.com/github/TheFinix13/NLP-coursework/blob/main/notebooks/2.2_RoBERTa_CrossVariety_Joel_Fiyin.ipynb)
- 2.3 LoRA: [notebooks/2.3_LoRA_Adapters_Mohamed.ipynb](https://colab.research.google.com/github/TheFinix13/NLP-coursework/blob/main/notebooks/2.3_LoRA_Adapters_Mohamed.ipynb)
- Deployment (Gradio): [notebooks/run_deployment_colab.ipynb](https://colab.research.google.com/github/TheFinix13/NLP-coursework/blob/main/notebooks/run_deployment_colab.ipynb)

The first code cell in each training notebook clones this repo and runs `pip install -r requirements.txt`. The BESSTIE dataset loads from Hugging Face: `surrey-nlp/BESSTIE-CW-26`.

To override the clone target (e.g. to test a fork), set `REPO_URL` and/or `REPO_BRANCH` env vars before running cell 1.
Notebooks **2.2** (RoBERTa) and **2.3** (LoRA) support **`DEMO_MODE`**: default is fast demo; set **`DEMO_MODE=0`** before running for full experiments (see checklist).

## Local setup
```bash
git clone https://github.com/TheFinix13/NLP-coursework.git
cd NLP-coursework
pip install -r requirements.txt
```

```python
from datasets import load_dataset
ds = load_dataset("surrey-nlp/BESSTIE-CW-26")
```
