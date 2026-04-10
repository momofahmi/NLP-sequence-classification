## Report template (fill-in)

This is a **template** to keep the final PDF report simple, readable, and aligned to the marking scheme.
Replace the TODO blocks with your own writing, tables, and screenshots from the notebooks.

---

### Title page
- **Module**: COMM061 Natural Language Processing
- **Coursework**: Group coursework (BESSTIE)
- **Group name**: TODO
- **Members**: TODO
- **Declaration of originality**: (attach per SurreyLearn requirement)

---

### 1. Dataset analysis and visualisation (max 4 pages)

#### 1.1 Label distributions (Sentiment / Sarcasm by variety)
- **What we plotted**: TODO (list charts)
- **Observations (class imbalance)**:
  - **Sarcasm**: TODO
  - **Sentiment**: TODO
- **Any additional interesting patterns** (source differences, split sizes, etc.): TODO

Include (screenshots or exported figures):
- Distribution plots for Sentiment across `en-UK/en-AU/en-IN`
- Distribution plots for Sarcasm across `en-UK/en-AU/en-IN`

#### 1.2 Vocabulary overlap + linguistic distance
- **Method**: TODO (e.g., Jaccard similarity on word sets; TF‑IDF cosine similarity on concatenated variety docs)
- **Results**:
  - **Inner-circle vs en-IN** (pick one): TODO (table of similarity scores)
- **Define “linguistic distance”**:
  - TODO: short definition (lexical + phraseology + code-mixing + pragmatics + orthography/grammar)
- **Interpretation**:
  - TODO: whether “distance” likely affects tokenization/domain shift and model transferability

Include:
- Heatmap of similarities
- 2–3 short examples of distinctive tokens/phrases per variety (no dataset dumping)

---

### 2. Experiments (max 6 pages)

General experimental settings (one short paragraph):
- **Dataset**: `surrey-nlp/BESSTIE-CW-26` (fixed train/val/test splits)
- **Primary metric**: Macro-F1
- **Seeds / repeated runs**: TODO (e.g., seeds 42 and 123)
- **Imbalance handling**: TODO (class weights, etc.)

#### 2.1 Baseline vs PTLM gap (10 marks)
**Task(s)**: TODO (Sarcasm / Sentiment)

- **Baseline**:
  - Features: TODO (TF‑IDF / GloVe)
  - Classifier: TODO (LogReg / SVM)
  - Results: TODO (macro-F1 + per-class precision/recall/F1)
- **Transformer**:
  - Model: TODO (RoBERTa-base)
  - Fine-tuning details: TODO (epochs, batch size, lr, max length)
  - Results: TODO (macro-F1 + per-class precision/recall/F1)
- **Analysis**:
  - TODO: why pretraining helps/doesn’t (domain shift, pragmatics in sarcasm, etc.)

Include:
- Table comparing baseline vs RoBERTa (per-class metrics, macro-F1)
- Confusion matrix for best model

#### 2.2 Cross-variety evaluation matrix (15 marks)
**Task**: TODO (pick one required by spec; optionally provide both as extra)

- **Protocol**:
  - Train on each variety separately; test on all three
  - No test-set modifications
- **Matrix results**:
  - Macro‑F1 matrix: TODO (3×3 table)
  - Per-class results: TODO (brief table or highlight worst-off class)
- **Analysis**:
  - TODO: transferability patterns (e.g., AU→UK vs AU→IN)
  - TODO: connect to “linguistic distance” and dataset sources

Include:
- Heatmap of cross-variety macro‑F1
- Confusion matrices for best 1–2 settings

#### 2.3 LLM variations with LoRA adapters (15 marks)
**Task**: Sarcasm

- **Base model (frozen)**: TODO (open-weight 1–3B)
- **Adapters**: one per variety (UK/AU/IN)
- **Training setup**:
  - TODO (LoRA rank r, alpha, dropout, target modules, epochs, lr)
  - Any reduced train size: TODO
- **Results**:
  - Compare adapters across each test set: TODO (table)
- **Analysis**:
  - TODO: does matching adapter→variety help?
  - TODO: does inner-circle adapter transfer better than to en-IN?

Include:
- Adapter-vs-test performance table
- Confusion matrix for best adapter+test pair

---

### 3. Evaluation (max 5 pages)

For each best model you highlight (baseline, RoBERTa, LoRA):
- Macro‑F1: TODO
- Per-class precision/recall/F1: TODO
- Confusion matrix: TODO

Short discussion:
- Where does the model fail (class imbalance / domain shift / slang / pragmatics): TODO

---

### 4. Error analysis + few-shot prompt (max 4 pages)

- **Select 10 errors** from best LLM/adapter: TODO (show text snippets, true vs pred; keep short)
- **Choose 4** and explain sarcasm/non-sarcasm with linguistic reasoning: TODO
- **Construct a 4-shot prompt** (include your explanations): TODO
- **Re-test remaining 6** with the few-shot prompt:
  - Before vs after: TODO
- **Conclusion**: TODO (what improved, what did not, and why)

---

### 5. Deployment + efficiency

#### 5.1 Web app / endpoint (max 5 pages)
- **Serving method**: TODO (Streamlit/Flask/Gradio)
- **UI requirement**:
  - text input
  - explicit variety selector (UK/AU/IN)
  - model/adapters switch based on selector
- **Why this approach**: TODO (simplicity, adapter swapping, etc.)
- Screenshots: TODO

#### 5.2 Efficiency (max 1 page)
- Compare inference time for 2 models (e.g., baseline vs RoBERTa vs LoRA):
  - small input: TODO ms
  - larger input/batch: TODO ms
- Discussion: TODO (size vs latency trade-off)

---

### References
Include dataset + any libraries/papers used.

