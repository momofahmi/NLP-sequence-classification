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

Note: To train on a GPU in Windows, install the CUDA version of PyTorch using:
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
