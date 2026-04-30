#!/usr/bin/env bash
# Build the SurreyLearn code-submission ZIP.
#
# Per the coursework brief:
#   - Code ZIP only — the PDF is uploaded separately.
#   - DO NOT include datasets, trained models, /runs, /checkpoints.
#   - DO include notebooks, helper modules, requirements.txt, README.
#
# Usage:
#   ./scripts/build_submission_zip.sh PG07            # pass your group code
#
# Produces: dist/PG07_code.zip
set -euo pipefail

GROUP_CODE="${1:-GROUP}"
OUT_DIR="dist"
ZIP_NAME="${OUT_DIR}/${GROUP_CODE}_code.zip"

mkdir -p "$OUT_DIR"
rm -f "$ZIP_NAME"

# Anything matching these patterns is excluded.
EXCLUDES=(
    # Git / IDE / OS junk
    '*/.git/*'      '.git/*'
    '*/.github/*'   '.github/*'
    '*/.idea/*'     '.idea/*'
    '*/.vscode/*'   '.vscode/*'
    '*/.DS_Store'   '.DS_Store'

    # Python caches
    '*/__pycache__/*'   '__pycache__/*'
    '*.pyc'
    '*/.ipynb_checkpoints/*'

    # Build / venv
    '.venv/*'   'venv/*'
    'dist/*'
    '.cache/*'

    # Notebook outputs that contain large embedded images / state
    # (we don't strip them automatically — but we DO strip widget metadata via sanitize_notebook.py)

    # Trained models / checkpoints / runs (brief: must NOT be in ZIP)
    'adapters/*/checkpoint-*/*'
    '*/checkpoint-*/*'
    'runs/*'
    'outputs/*'
    '*.bin'
    '*.safetensors'
    '*.pt'
    '*.pth'

    # Pickled classifiers / vectorizers — keep small ones, exclude very large ones
    'notebooks/models/multi_output_svm.pkl'
    'notebooks/models/separate_svm_*.pkl'
    'notebooks/models/tfidf/X_*_tfidf.npz'
    'models/tfidf/tfidf_vectorizer.pkl'

    # Tokenised dataset cache (regenerable; large .arrow files)
    'notebooks/tokenized/*'

    # Raw / processed data (brief: do NOT submit dataset)
    'data/*'
    '*.parquet'
    '*.csv'

    # PDFs from the docs folder (the report PDF is uploaded separately)
    '*.pdf'
)

# Build -x args for zip
ZIP_X_ARGS=()
for pat in "${EXCLUDES[@]}"; do
    ZIP_X_ARGS+=("-x" "$pat")
done

# Zip everything not excluded.
zip -r "$ZIP_NAME" . "${ZIP_X_ARGS[@]}"

echo
echo "Created: $ZIP_NAME"
echo "Size:    $(du -sh "$ZIP_NAME" | awk '{print $1}')"
echo
echo "Sanity check — files in the zip:"
unzip -l "$ZIP_NAME" | head -30
echo "  ..."
unzip -l "$ZIP_NAME" | tail -5
