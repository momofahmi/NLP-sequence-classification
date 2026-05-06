# Mohamed Fahmi Ahmed 

import gradio as gr
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE_MODEL = "facebook/opt-1.3b"
ADAPTERS = {
    "en-UK": "momofahmi/besstie-lora-en-uk-opt-1.3b",
    "en-AU": "momofahmi/besstie-lora-en-au-opt-1.3b",
    "en-IN": "momofahmi/besstie-lora-en-in-opt-1.3b",
}

ROBERTA_SARCASM   = "joela0/besstie-roberta-all-pool"
ROBERTA_SENTIMENT = "joela0/besstie-roberta-sentiment-all-pool"

MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load OPT base + 3 LoRA adapters 
print(f"Loading OPT base model on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=2, dtype=dtype,
)
base_model.config.pad_token_id = tokenizer.pad_token_id

print("Loading LoRA adapters...")
peft_model = PeftModel.from_pretrained(base_model, ADAPTERS["en-UK"], adapter_name="en-UK")
peft_model.load_adapter(ADAPTERS["en-AU"], adapter_name="en-AU")
peft_model.load_adapter(ADAPTERS["en-IN"], adapter_name="en-IN")
peft_model.eval()
peft_model = peft_model.to(DEVICE)

# load RoBERTa models 
print("Loading RoBERTa sarcasm model (all-pool)...")
rob_sarc_tok = AutoTokenizer.from_pretrained(ROBERTA_SARCASM)
rob_sarc_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_SARCASM).to(DEVICE)
rob_sarc_model.eval()

print("Loading RoBERTa sentiment model (all-pool)...")
rob_sent_tok = AutoTokenizer.from_pretrained(ROBERTA_SENTIMENT)
rob_sent_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_SENTIMENT).to(DEVICE)
rob_sent_model.eval()

print("Ready.")


# inference 
def predict_opt(text: str, variety: str):
    """OPT-1.3B + LoRA adapter (variety-specific) — sarcasm only."""
    peft_model.set_adapter(variety)
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, padding="max_length", max_length=MAX_LENGTH,
    ).to(DEVICE)
    with torch.no_grad():
        logits = peft_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    label = "Sarcastic" if probs[1] > 0.5 else "Not Sarcastic"
    return label, max(probs)


def predict_roberta_sarcasm(text: str):
    """RoBERTa (pooled across varieties) — sarcasm only."""
    inputs = rob_sarc_tok(
        text, return_tensors="pt",
        truncation=True, padding="max_length", max_length=MAX_LENGTH,
    ).to(DEVICE)
    with torch.no_grad():
        logits = rob_sarc_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    # BESSTIE Sarcasm convention: 0 = Not Sarcastic, 1 = Sarcastic
    label = "Sarcastic" if probs[1] > 0.5 else "Not Sarcastic"
    return label, max(probs)


def predict_roberta_sentiment(text: str):
    """RoBERTa (pooled across varieties) — sentiment only."""
    inputs = rob_sent_tok(
        text, return_tensors="pt",
        truncation=True, padding="max_length", max_length=MAX_LENGTH,
    ).to(DEVICE)
    with torch.no_grad():
        logits = rob_sent_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    # BESSTIE Sentiment convention: 0 = Negative, 1 = Positive
    label = "Positive" if probs[1] > 0.5 else "Negative"
    return label, max(probs)


def cell(label, conf):
    return f"{label} ({conf:.0%})"


def verdict_sarcasm(opt_preds, rob_sarc_label):
    """Verdict combining 3 OPT adapters + RoBERTa for sarcasm."""
    all_labels = [p[0] for p in opt_preds.values()] + [rob_sarc_label]
    sarc_count = all_labels.count("Sarcastic")
    total = len(all_labels)
    if sarc_count == total: return "All sarcastic"
    if sarc_count == 0:     return "All not sarcastic"
    return f"{sarc_count} of {total} sarcastic"


# mode 1: Single text with explicit variety
def analyze_single(text: str, variety: str):
    """User picks variety for OPT adapter that does sarcasm. RoBERTa does sentiment."""
    if not text.strip():
        return "", "", ""

    text = text.strip()

    # Sarcasm — OPT adapter (variety-specific)
    opt_lbl, opt_conf = predict_opt(text, variety)
    sarc_md = (
        f"### Sarcasm prediction\n"
        f"**{opt_lbl}** (confidence {opt_conf:.1%})\n\n"
        f"*Model: OPT-1.3B with LoRA adapter for {variety}*"
    )

    # Cross-check: does the second sarcasm model (RoBERTa) agree with OPT?
    rob_sarc_lbl, rob_sarc_conf = predict_roberta_sarcasm(text)
    if rob_sarc_lbl == opt_lbl:
        cross = (
            f"**Cross-check:** OPT {variety} and RoBERTa-pooled "
            f"agree on sarcasm ({opt_lbl})"
        )
    else:
        cross = (
            f"**Cross-check:** OPT {variety} ({opt_lbl}) and RoBERTa-pooled "
            f"({rob_sarc_lbl}) disagree on sarcasm"
        )

    # Sentiment — RoBERTa (variety-agnostic)
    sent_lbl, sent_conf = predict_roberta_sentiment(text)
    sent_md = (
        f"### Sentiment prediction\n"
        f"**{sent_lbl}** (confidence {sent_conf:.1%})\n\n"
        f"*Model: RoBERTa fine-tuned on all 3 varieties (pooled)*"
    )

    return sarc_md, sent_md, cross


# mode 2: Batch comparison across all models 
def analyze_batch(texts_input: str):
    """For each text, run all 5 models. Output is a comparison table."""
    texts = [t.strip() for t in texts_input.split("\n") if t.strip()]
    if not texts:
        return pd.DataFrame(), ""

    rows = []
    n_all_sarc, n_all_gen, n_disagree = 0, 0, 0

    for text in texts:
        preview = text if len(text) <= 60 else text[:57] + "..."

        # 3 OPT adapters (sarcasm, per-variety)
        opt_preds = {}
        for variety in ["en-UK", "en-AU", "en-IN"]:
            opt_preds[variety] = predict_opt(text, variety)

        # RoBERTa sarcasm + sentiment (pooled across varieties)
        rob_sarc_lbl, rob_sarc_conf = predict_roberta_sarcasm(text)
        rob_sent_lbl, rob_sent_conf = predict_roberta_sentiment(text)

        rows.append({
            "Text": preview,
            "OPT-1.3B en-UK (sarcasm)": cell(*opt_preds["en-UK"]),
            "OPT-1.3B en-AU (sarcasm)": cell(*opt_preds["en-AU"]),
            "OPT-1.3B en-IN (sarcasm)": cell(*opt_preds["en-IN"]),
            "RoBERTa pooled (sarcasm)": cell(rob_sarc_lbl, rob_sarc_conf),
            "RoBERTa pooled (sentiment)": cell(rob_sent_lbl, rob_sent_conf),
            "Sarcasm verdict": verdict_sarcasm(opt_preds, rob_sarc_lbl),
        })

        all_sarc_labels = [p[0] for p in opt_preds.values()] + [rob_sarc_lbl]
        sarc_n = all_sarc_labels.count("Sarcastic")
        if sarc_n == 4:   n_all_sarc += 1
        elif sarc_n == 0: n_all_gen += 1
        else:             n_disagree += 1

    n = len(texts)
    summary = (
        f"**{n}** text{'s' if n>1 else ''} analyzed across 5 models · "
    )
    return pd.DataFrame(rows), summary


# interface 
with gr.Blocks(title="BESSTIE Sarcasm Lab") as demo:

    gr.Markdown("# BESSTIE Sarcasm Lab")
    gr.Markdown(
        "Sarcasm and sentiment classification on the BESSTIE benchmark.\n\n"
        "- **Sarcasm**: 3 OPT-1.3B LoRA adapters (one per English variety) + 1 RoBERTa pooled across varieties\n"
        "- **Sentiment**: 1 RoBERTa pooled across varieties"
    )

    with gr.Tab("Single text"):
        gr.Markdown(
            "Enter a text and choose an English variety. "
            "The matching OPT adapter classifies sarcasm; "
            "RoBERTa classifies sentiment and provides a cross-check on sarcasm."
        )

        text_in = gr.Textbox(
            label="Text",
            lines=3,
            placeholder="Enter a sentence to classify...",
        )
        variety_in = gr.Radio(
            label="English variety (used by OPT adapter for sarcasm)",
            choices=["en-UK", "en-AU", "en-IN"],
            value="en-UK",
        )
        single_btn = gr.Button("Analyze", variant="primary")

        sarc_out  = gr.Markdown()
        sent_out  = gr.Markdown()
        cross_out = gr.Markdown()

        single_btn.click(
            fn=analyze_single,
            inputs=[text_in, variety_in],
            outputs=[sarc_out, sent_out, cross_out],
        )

    with gr.Tab("Compare all models"):
        gr.Markdown(
            "Enter one or more texts (one per line). Each text is classified by "
            "all 5 models: 3 OPT LoRA adapters (sarcasm), RoBERTa-pooled (sarcasm), "
            "and RoBERTa-pooled (sentiment)."
        )

        texts_in = gr.Textbox(
            label="Texts (one per line)",
            lines=6,
        )
        batch_btn = gr.Button("Analyze with all models", variant="primary")

        batch_summary = gr.Markdown()
        batch_table = gr.Dataframe(wrap=True, interactive=False)

        batch_btn.click(
            fn=analyze_batch,
            inputs=[texts_in],
            outputs=[batch_table, batch_summary],
        )

if __name__ == "__main__":
    demo.launch()
