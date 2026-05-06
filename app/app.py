# Mohamed Fahmi Ahmed

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from _compat import ensure_peft_compat
ensure_peft_compat()

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
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading base model on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=2, dtype=dtype,
)
base_model.config.pad_token_id = tokenizer.pad_token_id

print("Loading adapters...")
peft_model = PeftModel.from_pretrained(base_model, ADAPTERS["en-UK"], adapter_name="en-UK")
peft_model.load_adapter(ADAPTERS["en-AU"], adapter_name="en-AU")
peft_model.load_adapter(ADAPTERS["en-IN"], adapter_name="en-IN")
peft_model.eval()
peft_model = peft_model.to(DEVICE)
print("Ready.")


def predict(text: str, variety: str):
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

def cell(label, conf):
    return f"{label} ({conf:.0%})"

def verdict(preds):
    sarc = [p[0] for p in preds.values()].count("Sarcastic")
    if sarc == 3:  return "All sarcastic"
    if sarc == 0:  return "None is sarcastic"
    return f"{sarc} of 3 sarcastic"


# mode 1: Explicit selection 
def analyze_single(text: str, variety: str):
    if not text.strip():
        return "", ""

    label, conf = predict(text.strip(), variety)
    result = f"### Prediction: {label}\n\nConfidence: {conf:.1%}"
    detail = f"Adapter used: OPT-1.3B {variety}"
    return result, detail


# mode 2: Compare all adapters
def analyze_batch(texts_input: str):
    texts = [t.strip() for t in texts_input.split("\n") if t.strip()]
    if not texts:
        return pd.DataFrame(), ""

    rows = []
    n_all_sarc, n_all_gen, n_disagree = 0, 0, 0

    for text in texts:
        preview = text if len(text) <= 60 else text[:57] + "..."
        preds = {}
        for variety in ["en-UK", "en-AU", "en-IN"]:
            preds[variety] = predict(text, variety)

        rows.append({
            "Text": preview,
            "OPT-1.3B en-UK": cell(*preds["en-UK"]),
            "OPT-1.3B en-AU": cell(*preds["en-AU"]),
            "OPT-1.3B en-IN": cell(*preds["en-IN"]),
            "Verdict": verdict(preds),
        })

        sarc = [p[0] for p in preds.values()].count("Sarcastic")
        if sarc == 3: n_all_sarc += 1
        elif sarc == 0: n_all_gen += 1
        else: n_disagree += 1

    n = len(texts)
    summary = (
        f"**{n}** text{'s' if n>1 else ''} analyzed · "
        f"All sarcastic: {n_all_sarc} · "
        f"All genuine: {n_all_gen} · "
        f"Disagreement: {n_disagree}"
    )
    return pd.DataFrame(rows), summary


with gr.Blocks(title="BESSTIE Sarcasm Lab") as demo:

    gr.Markdown("# BESSTIE Sarcasm Lab")
    gr.Markdown(
        "Three OPT-1.3B LoRA adapters classify sarcasm : one trained per "
        "English variety (British, Australian, Indian Reddit data)."
    )

    with gr.Tab("Sarcasm Detection"):
        gr.Markdown(
            "Enter a text and choose an English variety. The backend "
            "loads the matching adapter to make the prediction."
        )

        text_in = gr.Textbox(
            label="Text",
            lines=3,
            placeholder="Enter a sentence to classify...",
        )
        variety_in = gr.Radio(
            label="English variety",
            choices=["en-UK", "en-AU", "en-IN"],
            value="en-UK",
        )
        single_btn = gr.Button("Analyze", variant="primary")

        single_result = gr.Markdown()
        single_detail = gr.Markdown()

        single_btn.click(
            fn=analyze_single,
            inputs=[text_in, variety_in],
            outputs=[single_result, single_detail],
        )

    with gr.Tab("Compare all adapters"):
        gr.Markdown(
            "Enter one or more texts (one per line). Each text is "
            "classified by all three adapters."
        )

        texts_in = gr.Textbox(
            label="Texts (one per line)",
            lines=6,
        )
        batch_btn = gr.Button("Analyze with all 3 adapters", variant="primary")

        batch_summary = gr.Markdown()
        batch_table = gr.Dataframe(wrap=True, interactive=False)

        batch_btn.click(
            fn=analyze_batch,
            inputs=[texts_in],
            outputs=[batch_table, batch_summary],
        )

if __name__ == "__main__":
    demo.launch()
