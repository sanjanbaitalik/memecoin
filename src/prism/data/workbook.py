from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from prism.utils.paths import find_project_root, output_path

REQUIRED_BASE_COLUMNS = [
    "token",
    "ticker",
    "date",
    "price",
    "volume",
]

OPTIONAL_SOCIAL_COLUMNS = [
    "twitter_sentiments",
    "telegram_sentiment",
    "reddit_sentiment",
]


@dataclass
class TokenSheetSummary:
    sheet_name: str
    rows: int
    token: str
    ticker: str
    first_date: str | None
    last_date: str | None
    valid_schema: bool
    missing_required_columns: str


def locate_workbook() -> Path:
    root = find_project_root()
    override = os.environ.get("PRISM_INPUT_WORKBOOK", "").strip()
    if override:
        override_path = Path(override)
        if override_path.exists():
            return override_path

    candidates = [
        root / "data" / "raw" / "Output_database.xlsx",
        root / "Output_database.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    globbed = list((root / "data" / "raw").glob("*.xlsx"))
    if globbed:
        return globbed[0]

    raise FileNotFoundError(
        "Workbook not found. Expected Output_database.xlsx in data/raw or project root."
    )


def list_sheets(workbook_path: Path) -> list[str]:
    excel = pd.ExcelFile(workbook_path)
    return excel.sheet_names


def read_sheet(workbook_path: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(workbook_path, sheet_name=sheet_name)


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [str(c).strip().lower() for c in renamed.columns]
    return renamed


def _is_meaningful_time_column(time_series: pd.Series) -> bool:
    non_null = time_series.dropna()
    if non_null.empty:
        return False

    # If time is numeric and almost always zero, date-only granularity should be preserved.
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().mean() > 0.9 and (numeric.fillna(0).abs() < 1e-12).mean() > 0.95:
        return False

    as_str = non_null.astype(str).str.strip()
    return bool((as_str.str.contains(":", regex=False)).mean() > 0.2 or numeric.notna().mean() > 0.9)


def coerce_datetime_with_diagnostics(
    df: pd.DataFrame,
    sheet_name: str,
    diagnostics: list[dict] | None = None,
) -> pd.Series:
    if "date" not in df.columns:
        if diagnostics is not None:
            diagnostics.append(
                {
                    "sheet_name": sheet_name,
                    "issue": "missing_date_column",
                    "date_dtype": "",
                    "date_sample": "",
                }
            )
        return pd.Series([pd.NaT] * len(df), index=df.index)

    date_raw = pd.to_datetime(df["date"], errors="coerce")

    if "time" not in df.columns or not _is_meaningful_time_column(df["time"]):
        if diagnostics is not None and date_raw.notna().sum() == 0:
            sample = df["date"].dropna().astype(str).head(3).tolist()
            diagnostics.append(
                {
                    "sheet_name": sheet_name,
                    "issue": "date_parse_failed",
                    "date_dtype": str(df["date"].dtype),
                    "date_sample": " | ".join(sample),
                }
            )
        return date_raw

    combined = pd.to_datetime(
        df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
        errors="coerce",
    )

    # Guardrail: do not destroy valid parsed dates due to malformed time column.
    if combined.notna().sum() < date_raw.notna().sum() * 0.5:
        return date_raw

    if diagnostics is not None and combined.notna().sum() == 0:
        diagnostics.append(
            {
                "sheet_name": sheet_name,
                "issue": "datetime_merge_failed",
                "date_dtype": str(df["date"].dtype),
                "date_sample": " | ".join(df["date"].dropna().astype(str).head(3).tolist()),
            }
        )

    return combined


def load_workbook_frames(workbook_path: Path) -> tuple[list[str], dict[str, pd.DataFrame], list[dict]]:
    frames: dict[str, pd.DataFrame] = {}
    read_errors: list[dict] = []
    xl = pd.ExcelFile(workbook_path)
    for sheet in xl.sheet_names:
        try:
            frame = pd.read_excel(xl, sheet_name=sheet)
            frames[sheet] = _standardize_columns(frame)
        except Exception as exc:
            read_errors.append({"sheet_name": sheet, "error": f"{type(exc).__name__}: {exc}"})
    return xl.sheet_names, frames, read_errors


def summarize_sheet(sheet_name: str, frame: pd.DataFrame) -> TokenSheetSummary:
    columns_lower = {str(c).strip().lower(): c for c in frame.columns}
    missing_required = [c for c in REQUIRED_BASE_COLUMNS if c not in columns_lower]

    diagnostics: list[dict] = []
    dt = coerce_datetime_with_diagnostics(
        frame.rename(columns={v: k for k, v in columns_lower.items()}),
        sheet_name=sheet_name,
        diagnostics=diagnostics,
    )

    token_col = columns_lower.get("token")
    ticker_col = columns_lower.get("ticker")
    token = str(frame[token_col].dropna().iloc[0]) if token_col and not frame[token_col].dropna().empty else sheet_name
    ticker = str(frame[ticker_col].dropna().iloc[0]) if ticker_col and not frame[ticker_col].dropna().empty else ""

    first_date = dt.min()
    last_date = dt.max()

    return TokenSheetSummary(
        sheet_name=sheet_name,
        rows=len(frame),
        token=token,
        ticker=ticker,
        first_date=first_date.isoformat() if pd.notna(first_date) else None,
        last_date=last_date.isoformat() if pd.notna(last_date) else None,
        valid_schema=not missing_required,
        missing_required_columns=";".join(missing_required),
    )


def audit_workbook(workbook_path: Path) -> dict[str, pd.DataFrame | dict]:
    sheets, frames, read_errors = load_workbook_frames(workbook_path)
    inventory_rows: list[dict] = []
    missingness_rows: list[dict] = []
    schema_columns: dict[str, int] = {}
    date_parse_diagnostics: list[dict] = []

    for sheet in sheets:
        frame = frames.get(sheet)
        if frame is None:
            inventory_rows.append(
                {
                    "sheet_name": sheet,
                    "rows": 0,
                    "token": sheet,
                    "ticker": "",
                    "first_date": None,
                    "last_date": None,
                    "valid_schema": False,
                    "missing_required_columns": "sheet_read_error",
                }
            )
            continue

        summary = summarize_sheet(sheet, frame)
        inventory_rows.append(summary.__dict__)

        dt = coerce_datetime_with_diagnostics(frame, sheet_name=sheet, diagnostics=date_parse_diagnostics)
        valid_date_count = int(dt.notna().sum()) if not dt.empty else 0
        inventory_rows[-1]["valid_date_count"] = valid_date_count
        inventory_rows[-1]["date_parse_rate"] = float(valid_date_count / len(frame)) if len(frame) else 0.0

        for column in frame.columns:
            schema_columns[column] = schema_columns.get(column, 0) + 1
            missingness_rows.append(
                {
                    "sheet_name": sheet,
                    "column": column,
                    "missing_fraction": float(frame[column].isna().mean()),
                    "non_null_count": int(frame[column].notna().sum()),
                    "row_count": int(len(frame)),
                }
            )

    inventory_df = pd.DataFrame(inventory_rows)
    missingness_df = pd.DataFrame(missingness_rows)
    diagnostics_df = pd.DataFrame(date_parse_diagnostics)

    if not inventory_df.empty:
        no_date_fraction = float((inventory_df.get("valid_date_count", 0) == 0).mean())
        if no_date_fraction > 0.9:
            raise ValueError(
                f"Date parsing failed for {no_date_fraction:.1%} of sheets. Inspect outputs/audits/date_parse_diagnostics.csv"
            )

    schema_summary = {
        "sheet_count": len(sheets),
        "required_columns": REQUIRED_BASE_COLUMNS,
        "optional_social_columns": OPTIONAL_SOCIAL_COLUMNS,
        "column_presence_count": schema_columns,
        "valid_sheet_count": int(inventory_df["valid_schema"].sum()) if not inventory_df.empty else 0,
        "read_error_count": len(read_errors),
        "date_parse_issue_count": int(len(diagnostics_df)),
    }

    return {
        "inventory": inventory_df,
        "missingness": missingness_df,
        "date_parse_diagnostics": diagnostics_df,
        "schema_summary": schema_summary,
    }


def write_audit_outputs(audit: dict[str, pd.DataFrame | dict]) -> None:
    output_path("outputs", "audits")

    inventory = audit["inventory"]
    missingness = audit["missingness"]
    date_parse_diagnostics = audit.get("date_parse_diagnostics", pd.DataFrame())
    schema_summary = audit["schema_summary"]

    assert isinstance(inventory, pd.DataFrame)
    assert isinstance(missingness, pd.DataFrame)
    assert isinstance(schema_summary, dict)

    inventory.to_csv(output_path("outputs", "audits", "workbook_inventory.csv"), index=False)
    missingness.to_csv(output_path("outputs", "audits", "missingness_report.csv"), index=False)
    if isinstance(date_parse_diagnostics, pd.DataFrame):
        date_parse_diagnostics.to_csv(
            output_path("outputs", "audits", "date_parse_diagnostics.csv"),
            index=False,
        )
    output_path("outputs", "audits", "schema_summary.json").write_text(
        json.dumps(schema_summary, indent=2), encoding="utf-8"
    )


def iter_token_frames(workbook_path: Path, sheets: Iterable[str] | None = None) -> Iterable[tuple[str, pd.DataFrame]]:
    selected = set(sheets) if sheets is not None else None
    _, frames, _ = load_workbook_frames(workbook_path)
    for sheet, frame in frames.items():
        if selected is not None and sheet not in selected:
            continue
        yield sheet, frame
