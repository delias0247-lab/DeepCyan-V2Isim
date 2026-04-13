import re
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

def ensure_outputs_mount(app: FastAPI, outputs_root: Path):
    outputs_root.mkdir(parents=True, exist_ok=True)

    idx = outputs_root / "index.html"
    if not idx.exists():
        idx.write_text(
            """<!doctype html>
<html>
<head><meta charset="utf-8"><title>Outputs</title></head>
<body style="font-family:Arial;margin:24px">
  <h2>Outputs Root</h2>
  <p>This folder contains run outputs (CSV + plots) grouped by scenario.</p>
</body>
</html>""",
            encoding="utf-8",
        )

    app.mount("/outputs", StaticFiles(directory=str(outputs_root), html=True), name="outputs")

def make_run_dir(outputs_root: Path, scenario: str, tag: str) -> Path:
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(tag).stem).strip("_") or "run"
    run_id = datetime.now().strftime(f"run_%Y-%m-%d_%H%M%S_{safe_tag}")
    run_dir = outputs_root / scenario / run_id
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir

def write_run_index(run_dir: Path, title: str, subtitle: str, files: List[str], plots: List[str]):
    files_html = "\n".join(f'    <li><a href="{name}">{name}</a></li>' for name in files)
    plots_html = "\n".join(f'  <img src="{name}" alt="{name}"/>' for name in plots) or "<p>No plots generated.</p>"

    idx = run_dir / "index.html"
    idx.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{title} - {run_dir.name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    img {{ max-width: 1000px; display:block; margin: 12px 0; border:1px solid #ddd; }}
    code {{ background:#f4f4f4; padding:2px 6px; border-radius:4px; }}
  </style>
</head>
<body>
  <h1>{title} / {run_dir.name}</h1>
  <p><b>Model:</b> <code>{subtitle}</code></p>

  <h3>Files</h3>
  <ul>
{files_html}
  </ul>

  <h3>Plots</h3>
{plots_html}
</body>
</html>
""",
        encoding="utf-8",
    )
