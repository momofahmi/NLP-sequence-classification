# NLP-sequence-classification
**BESSTIE: Sentiment & Sarcasm Classification across English Varieties**
University of Surrey ,Semester 2, 2026

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

A `.env` file at the project root is required for HuggingFace authentication.

**`main.ipynb` must stay at the repository root.**

---

## Deployed app

The Gradio app combines all 5 models (3 LoRA adapters for sarcasm + 2 RoBERTa models for sarcasm and sentiment) and is publicly accessible at:

**https://huggingface.co/spaces/momofahmi/besstie-cw-nlp**

The app has two tabs: a single-text mode where the user explicitly selects an English variety (per the coursework spec), and a comparison mode that runs every text through all 5 models in parallel.

---
