#!/usr/bin/env python3
"""Remove Jupyter widget metadata that breaks GitHub notebook rendering.

Error: the 'state' key is missing from 'metadata.widgets'

Run before commit if you saved from Colab with tqdm/ipwidgets:
  python3 scripts/sanitize_notebook.py notebooks/2.2_RoBERTa_CrossVariety_Joel_Fiyin.ipynb
  python3 scripts/sanitize_notebook.py notebooks/*.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def sanitize_notebook(nb: dict) -> dict:
    if isinstance(nb.get("metadata"), dict):
        nb["metadata"].pop("widgets", None)

    for cell in nb.get("cells", []):
        if isinstance(cell.get("metadata"), dict):
            cell["metadata"].pop("widgets", None)

        outs = cell.get("outputs")
        if not outs:
            continue
        new_outs: list = []
        for out in outs:
            if not isinstance(out, dict):
                new_outs.append(out)
                continue
            om = out.get("metadata")
            if isinstance(om, dict):
                om.pop("widgets", None)
            data = out.get("data")
            if isinstance(data, dict):
                for mime in list(data.keys()):
                    if "jupyter.widget" in mime or "application/vnd.jupyter.widget" in mime:
                        del data[mime]
            if out.get("output_type") == "display_data" and isinstance(data, dict) and len(data) == 0:
                continue
            new_outs.append(out)
        cell["outputs"] = new_outs
    return nb


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("Usage: sanitize_notebook.py <notebook.ipynb> [...]", file=sys.stderr)
        sys.exit(1)
    for path in paths:
        if path.is_dir():
            paths.extend(sorted(path.glob("*.ipynb")))
            continue
        if not path.suffix == ".ipynb":
            continue
        text = path.read_text(encoding="utf-8")
        nb = json.loads(text)
        sanitize_notebook(nb)
        path.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
        print("OK", path)


if __name__ == "__main__":
    main()
