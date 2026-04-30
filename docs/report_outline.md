# Final report — canonical outline & Google Docs formatting guide

> Submission file: `report_PG##.pdf` (replace `##` with the group number).
> Hard caps from the brief: **25 pages of content** (excluding refs and appendix), plus a title page with declaration of originality.

This file is the structural source-of-truth. The Google Doc at <https://docs.google.com/document/d/1tgx2yC--QY4OnMiS2L-xbYXM3UQQ6u7ChX-118BzdIM> should match this outline section-for-section before the team exports to PDF.

---

## Page budget (firm — keep 1-page slack for figures)

| Section | Marks | Max pages | Soft target |
|---|---:|---:|---:|
| Title + declaration | — | 1 | 1 |
| §1 Data analysis (1.1 + 1.2) | 15 | 4 | 3.5 |
| §2 Experimentation (2.1 + 2.2 + 2.3) | 40 | 6 | 6 |
| §3 Evaluation | 15 | 5 | 4.5 |
| §4 Error analysis & few-shot | 10 | 4 | 3.5 |
| §5.1 Deployment endpoint | 15 | 5 | 4 |
| §5.2 Efficiency | 5 | 1 | 1 |
| References | — | unlimited | 1 |
| **Total content** | **100** | **25** | **23.5** |

If we exceed 25 pages of content, the over-spill section to trim first is **§2.3 LoRA** (Mohamed's draft is detailed and can lose ~half a page on Study 1 ablation table commentary), then **§4** discussion paragraphs.

---

## Section-by-section outline

Each section lists: (a) the heading exactly as it should appear in the doc, (b) which prose file in `reports/results/` is the source, (c) which figures/tables to include, (d) the owner, and (e) what's still TODO.

### Title page (page 1)

- Group code & name
- Member names + URNs
- Module: COMM061 — Natural Language Processing — Group Coursework
- Submission date: 6 May 2026
- **Declaration of originality** (template on My Surrey → Exams and Assessments → Assessments) — every member signs (digital signature, scanned image, or Adobe sign feature).

### §1 — Data analysis and visualisation (15 marks, max 4 pages)

#### 1.1 Distribution and class imbalance (5 marks)
- **Source:** existing prose in the docx (Yusrah). Already complete.
- **Figures:** `reports/figures/q1_1_*.png` — pick the strongest 4–5:
  - variety distribution
  - sentiment by variety
  - sarcasm by variety (highlights the imbalance)
  - source × variety stacked bar
  - sarcasm-sentiment correlation cell that supports the 97.52% finding
- **Tables:** the 2 small tables Yusrah references (Tables 1, 2 — sentiment/sarcasm by source).

#### 1.2 Vocabulary analysis (10 marks)
- **Source:** `reports/results/q1_2_vocab_overlap.md` (full prose ready, paste).
- **Figure:** `notebooks/reports/figures/vocabulary_similarity_heatmap.png`.
- **Table:** Table 1.2.1 (Jaccard + TF-IDF cosine, three pairs).
- **Required paragraph from the brief:** definition of *linguistic distance* + comment on whether the variety gap is superficial or grammatical — already written.

### §2 — Experimentation (40 marks, max 6 pages)

#### 2.1 Baseline / PTLM gap (10 marks)
- **Source:** existing prose in the docx (Yusrah). Already complete; mostly polish.
- **Tables:** Table a (Test set performance — TF-IDF+LR for sentiment + sarcasm per variety + pooled).
- **Figures:** TF-IDF + LR Macro-F1 comparison bar chart vs RoBERTa pooled (a single chart is enough).
- **Action:** drop the un-referenced placeholder *"MACRO F1 COMPARISON NEEDED"* once Joel's RoBERTa numbers are in §2.2 — the §2.1 comparison numbers come from there.

#### 2.2 Cross-variety evaluation — RoBERTa (15 marks)
- **Source:** Joel's RoBERTa notebook (`task-2.2.ipynb`, currently on `origin/Joel`) + the cross-variety matrix figure.
- **Tables:**
  - Table 2.2.1 — variety-only 3×3 cross-variety Macro-F1 (mean over seeds 42, 123).
  - Table 2.2.2 — extended pools (`inner_pool`, `all`) — gives the comparison required by §2.1.
- **Figures:**
  - `weighted_figures/cross_variety_matrix.png` (heatmap).
  - `weighted_figures/confusion_matrix_best.png` (best-condition confusion matrix; cite again in §3).
- **Action:** Joel writes the 2-paragraph analysis (which condition wins each test column, transferability commentary). Use language in line with the BESSTIE paper finding (cross-variety degradation, en-IN hardest).

#### 2.3 LoRA adapters (15 marks)
- **Source:** existing prose in the docx (Mohamed). Already complete.
- **Tables:**
  - Table 2 (Cross-variety Macro-F1 ± std).
  - Table 3 (Sarcasm-class F1).
- **Figures:**
  - `results/opt1.3B/training_curves.png`.
  - `results/opt1.3B/cross_variety_matrices.png`.
  - `results/opt1.3B/ablation.png` (optional — only if Study 1 is kept in the main text rather than appendix).
- **Action:** trim Mohamed's Study 1 ablation discussion if total content exceeds 25 pages.

### §3 — Evaluation (15 marks, max 5 pages)

This section is **where the brief explicitly demands per-class precision/recall + confusion matrices for the best models of each setup**. Structure as three sub-sub-sections, one per model family.

#### 3.1 Classical baseline — Logistic Regression + TF-IDF
- **Source:** existing prose in the docx (Yusrah). Already complete.
- **Figures:** Sarcasm CM, Sentiment CM, Precision/Recall bar chart per class.

#### 3.2 RoBERTa — best cross-variety condition
- **Source:** Joel's `weighted_figures/confusion_matrix_best.png` + the per-class metrics from his JSON.
- **Action:** ~½ page of prose stating which row of the matrix won (likely `all` or `inner_pool`), per-class precision/recall, where it fails (en-IN sarcasm).

#### 3.3 LoRA — best adapter, in-variety
- **Source:** `results/opt1.3B/confusion_matrices.png` (3-panel, one per variety adapter on its own test set).
- **Action:** ~½ page of prose with per-class precision/recall pulled from Mohamed's notebook's classification report.

### §4 — Sarcasm explanation & error analysis (10 marks, max 4 pages)

- **Source:** `reports/results/q4_error_analysis.md` (template ready).
- **Owner:** Mohammad.
- **Workflow:**
  1. `python scripts/q4_extract_errors.py` → `reports/results/q4_errors.json`
  2. Write `explanation` for 4 of the 10 examples in the JSON.
  3. `python scripts/q4_few_shot_eval.py` → `reports/results/q4_fewshot_results.json`
  4. (Optional, recommended) `python scripts/lime_explain.py --model lora --in reports/results/q4_errors.json` → `reports/figures/lime/`.
- **Sub-sections:** 4.1 Errors table, 4.2 Linguistic explanations, 4.3 The 4-shot prompt, 4.4 Re-test results (before/after table), 4.5 Discussion, 4.6 (optional) LIME panels.

### §5 — Deployment & efficiency (20 marks)

#### 5.1 Endpoint (15 marks, max 5 pages)
- **Source:** `reports/results/q5_1_deployment.md` (full prose ready, paste).
- **Owner:** Mohamed (app already built); Fiyin/anyone can paste the prose.
- **Figures:** 3 screenshot placeholders flagged in the doc.
- **Action:** capture the 3 screenshots from the running app — see `app/README.md` for instructions.

#### 5.2 Efficiency (5 marks, max 1 page)
- **Source:** `reports/results/q5_2_efficiency.md` (template + table skeleton ready).
- **Action:** run `python scripts/benchmark_inference.py --tfidf-vec ... --tfidf-clf ... --roberta roberta-base --base-llm facebook/opt-1.3b --lora momofahmi/besstie-lora-en-uk-opt-1.3b` once on Colab T4, paste 9 numbers into Table 5.2.1.

### References (no page limit)

Numbered list, IEEE-or-APA — pick one and be consistent. Minimum set:

1. Srirag, D., Joshi, A., Painter, J., & Kanojia, D. (2025). *BESSTIE: A Benchmark for Sentiment and Sarcasm Classification for Varieties of English.* Findings of ACL 2025. arXiv:2412.04726.
2. Liu, Y. et al. (2019). *RoBERTa: A robustly optimized BERT pretraining approach.* arXiv:1907.11692.
3. Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
4. Dettmers, T. et al. (2023). *QLoRA: Efficient finetuning of quantized LLMs.* NeurIPS 2023.
5. Plank, B. (2022). *The 'Problem' of Human Label Variation.* EMNLP 2022.
6. Abercrombie, G., & Hovy, D. (2016). *Putting Sarcasm Detection into Context: The Effects of Class Imbalance and Manual Labelling on Supervised Machine Classification of Twitter Conversations.* ACL 2016 SRW.
7. Skalicky, S., & Crossley, S. (2018). *Linguistic Features of Sarcasm and Metaphor Production Quality.* FigLang 2018.
8. Joshi, A. et al. (2025). *Natural language processing for dialects of a language: A survey.* ACM Computing Surveys 57(6).
9. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why should I trust you?": Explaining the predictions of any classifier.* KDD 2016.  *(only if LIME is used)*
10. Kortmann, B. et al. (2020). *eWAVE.* Online resource.

### Appendix (optional, after references — does not count against 25 pages)

Use this for material that is referenced from the body but doesn't fit. Good candidates:

- LoRA Study 1 ablation table in full (if trimmed from §2.3).
- LLaMA-1B vs LLaMA-3.2-3B alternative LoRA bases summary.
- Additional LIME plots (ones not shown in §4.6).

---

## Google Docs formatting checklist

The Google Doc is the rendering target. To make it look "decent and formal", apply these settings *once*, in this order:

### 1. Page setup (File → Page setup)
- Page size: **A4** (the brief is from a UK university; A4 is safer than Letter).
- Margins: **2.0 cm** all sides (default Google value is fine).
- Orientation: Portrait.

### 2. Default styles (Format → Paragraph styles → Options → Save my current styles as my default)
- **Body text**: Arial 11 (or Calibri 11) — single spacing, justified.
- **Heading 1**: Arial 16 bold, "Space before 18 pt, after 6 pt", `1` numbering style. Used for **§1, §2, §3, §4, §5**.
- **Heading 2**: Arial 13 bold, "Space before 12 pt, after 4 pt". Used for **§1.1, §2.1**, etc.
- **Heading 3**: Arial 11 bold (italic optional), no extra spacing. Used for sub-sub-sections (§3.1, §4.6).
- **Caption**: Arial 9 italic, centred. Used under figures and tables.

To save the default: write one Heading 1 with the right size/spacing, click in it, then `Format → Paragraph styles → Heading 1 → Update Heading 1 to match`. Repeat for every style. Then `Format → Paragraph styles → Options → Save my current styles as my default`.

### 3. Title page (page 1)
- Centred block, vertically centred (use a few empty lines or a single-cell table set to "vertical align middle"):
  ```text
  Group: PG## — <Group Name>

  Members:
  Yusrah <surname> (URN xxxxxxx)
  Mohamed Fahmi Ahmed (URN xxxxxxx)
  Mohammad <surname> (URN xxxxxxx)
  Joel <surname> (URN xxxxxxx)
  Omar <surname> (URN xxxxxxx)
  Fiyin Akano (URN xxxxxxx)

  COMM061 — Natural Language Processing — Group Coursework
  6 May 2026
  ```
- Below the block, paste the **Declaration of Originality** from My Surrey, with each member's name + signature image.
- Insert a **page break** at the end (`Cmd+Enter` on Mac, `Ctrl+Enter` on Windows) so §1 starts at the top of page 2.

### 4. Page numbers (Insert → Page numbers)
- Pick the variant that **starts numbering on page 2** (so the title page is unnumbered) and shows the number in the footer.

### 5. Headings & numbering
- Use the **Heading 1 / Heading 2 / Heading 3** styles from the dropdown, **not** manual bold/large text. This is what generates the table of contents and the export-to-PDF bookmarks.
- Number sections manually as `1.`, `1.1`, `1.1.1` to match this outline (Google Docs has auto-numbering under `Format → Bullets & numbering → List options`, but it is finicky — manual numbering is more reliable).

### 6. Table of contents (optional but professional)
- After the title page, insert `Insert → Table of contents → "with page numbers"`. Right-click and "Update table of contents" before exporting to PDF. Keep it on its own page; it does **not** count against the 25-page content limit because it's metadata.

### 7. Figures
- Insert as `Insert → Image → Upload from computer`. Set **Wrap text → In line** for predictable layout.
- Caption format: directly below the figure, centred, italic, **9 pt**:
  ```
  Figure 1.1.3 — Sarcasm class distribution by English variety, train split (BESSTIE-CW-26).
  ```
- Number figures `<section>.<figure_number>` (e.g. Figure 2.2.1, Figure 4.6.1). Keeps them findable when teammates edit.

### 8. Tables
- Use **Insert → Table**, not screenshots. Header row: bold + light-grey background (`Table → Table properties → Cell background colour → 10% grey`).
- Right-align numeric columns, left-align text columns. Centre column headers.
- Caption *above* the table, same style as figure captions.

### 9. Code blocks (only inside §2.3 / §4.3 / §5.1 if needed)
- Use a **single-cell table** with grey background and the **Consolas** or **Courier New** font, 9 pt. This survives PDF export better than Google's "Code block" which sometimes mangles in print.

### 10. References
- Use **numbered list** (`1.`, `2.`, …) — match the in-text citation style.
- Cite in the body as `[1]`, `[1, 4]`, etc., or `(Srirag et al., 2025)` — pick one and apply throughout.

### 11. Final pre-export pass
- Turn on `View → Show ruler` and `View → Show non-printing characters` once to spot stray double spaces / extra blank lines.
- Run `Tools → Spelling and grammar` over the whole doc.
- `File → Download → PDF (.pdf)` and verify:
  - Title page is page 1, no page number.
  - Page numbers start on page 2 = §1.
  - All figures rendered (not greyed-out broken-image boxes).
  - Total content pages ≤ 25.
- Rename the downloaded file to `report_PG##.pdf` before uploading to SurreyLearn.

---

## Pre-submission sanity-check (the last 60 minutes)

Run this list immediately before submitting on 6 May:

1. `git pull` on the canonical branch (probably `main` after Joel is merged).
2. Re-run each canonical notebook cell once on a fresh Colab kernel — no errors, plots render. Notebooks: `NLP_EDA.ipynb`, `SVM_TFIDF.ipynb`, `task-2.2.ipynb` (Joel's), `lora_notebook_opt1.3B.ipynb`.
3. `./scripts/build_submission_zip.sh PG##` to produce `dist/PG##_code.zip`. Open it and verify there are no `.bin`, `.safetensors`, `.pt`, `.parquet`, `checkpoint-*/` files inside.
4. Export the Google Doc → `report_PG##.pdf`. Check title page and page numbering.
5. Upload **both files** to SurreyLearn (PDF separate from ZIP per the brief).
6. Take a screenshot of the SurreyLearn submission confirmation and post it in the group chat.
