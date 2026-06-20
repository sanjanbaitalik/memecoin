from __future__ import annotations

from pathlib import Path

import pandas as pd

from scie_revision.common import scie_path, save_table_bundle


BANNED_PHRASES = [
    "fallback",
    "surrogate",
    "proxy MAML",
    "meta-learning-inspired adaptation proxy",
    "sklearn_mlp",
]


def generate_style_check_report(report_dir: Path | None = None) -> pd.DataFrame:
    report_root = report_dir or scie_path("outputs", "scie_revision_round3", "reports")
    rows: list[dict] = []
    for md_path in report_root.glob("*.md"):
        if md_path.name == "style_check_report.md":
            continue
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        for phrase in BANNED_PHRASES:
            if phrase.lower() in text.lower():
                rows.append({"file": md_path.name, "phrase": phrase, "suggested_replacement": "remove or revise"})
    for md_path in (report_root / "manuscript_inserts").glob("*.md") if (report_root / "manuscript_inserts").exists() else []:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        for phrase in BANNED_PHRASES:
            if phrase.lower() in text.lower():
                rows.append({"file": f"manuscript_inserts/{md_path.name}", "phrase": phrase, "suggested_replacement": "remove or revise"})
    table = pd.DataFrame(rows, columns=["file", "phrase", "suggested_replacement"]) if rows else pd.DataFrame([{"file": "none", "phrase": "none", "suggested_replacement": "no issues found"}])
    save_table_bundle(table, scie_path("outputs", "scie_revision_round3", "reports", "style_check_report"), "Style Check Report", "Automated check for banned phrases in manuscript inserts.")
    return table
