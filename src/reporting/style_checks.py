from __future__ import annotations

from pathlib import Path

import pandas as pd

from scie_revision.common import scie_path, save_table_bundle


BANNED_PHRASES = [
    "has been projected",
    "has been anticipated",
    "has been introduced",
    "statistical implication has been abridged",
    "existing forecasting research often treat",
    "baseline characteristics are used to analyze are shown",
    "persistent gaps such as existing approaches",
]


def generate_style_check_report(report_dir: Path | None = None) -> pd.DataFrame:
    report_root = report_dir or scie_path("outputs", "scie_revision", "reports")
    rows: list[dict] = []
    for md_path in report_root.glob("*.md"):
        if md_path.name == "style_check_report.md":
            continue
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        for phrase in BANNED_PHRASES:
            if phrase in text:
                rows.append({"file": md_path.name, "phrase": phrase, "suggested_replacement": "revise for clarity and tense consistency"})
    table = pd.DataFrame(rows, columns=["file", "phrase", "suggested_replacement"])
    save_table_bundle(table if not table.empty else pd.DataFrame([{"file": "none", "phrase": "none", "suggested_replacement": "no issues found"}]), scie_path("outputs", "scie_revision", "reports", "style_check_report"), "Style Check Report", "Automated wording check for generated manuscript snippets.")
    return table
