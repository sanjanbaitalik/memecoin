from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class DataContractError(RuntimeError):
    """Raised when an expected intermediate artifact is missing or invalid."""


@dataclass
class ContractContext:
    step: str
    path: Path
    expected_schema: list[str] | None = None
    upstream_dependency: str | None = None

    def message(self, detail: str) -> str:
        expected = f" expected_schema={self.expected_schema}" if self.expected_schema else ""
        upstream = f" upstream_dependency={self.upstream_dependency}" if self.upstream_dependency else ""
        return f"[{self.step}] {detail} path={self.path}{expected}{upstream}"


def safe_read_csv(
    path: Path,
    *,
    step: str,
    required_columns: list[str] | None = None,
    upstream_dependency: str | None = None,
) -> pd.DataFrame:
    ctx = ContractContext(
        step=step,
        path=path,
        expected_schema=required_columns,
        upstream_dependency=upstream_dependency,
    )

    if not path.exists():
        raise DataContractError(ctx.message("file does not exist"))
    if path.stat().st_size <= 0:
        raise DataContractError(ctx.message("file is zero bytes"))

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DataContractError(ctx.message(f"read_csv failed: {type(exc).__name__}: {exc}")) from exc

    if df.empty:
        raise DataContractError(ctx.message("dataframe is empty"))
    if len(df.columns) == 0:
        raise DataContractError(ctx.message("dataframe has zero columns"))

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise DataContractError(ctx.message(f"missing required columns: {missing}"))

    return df


def safe_write_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    step: str,
    allow_empty: bool = False,
) -> Path:
    if (df.empty or len(df.columns) == 0) and not allow_empty:
        raise DataContractError(f"[{step}] refusing to write empty dataframe to path={path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    if path.stat().st_size <= 0:
        raise DataContractError(f"[{step}] wrote zero-byte csv path={path}")
    return path


def safe_read_parquet(
    path: Path,
    *,
    step: str,
    required_columns: list[str] | None = None,
    upstream_dependency: str | None = None,
) -> pd.DataFrame:
    ctx = ContractContext(
        step=step,
        path=path,
        expected_schema=required_columns,
        upstream_dependency=upstream_dependency,
    )

    if not path.exists():
        raise DataContractError(ctx.message("file does not exist"))
    if path.stat().st_size <= 0:
        raise DataContractError(ctx.message("file is zero bytes"))

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise DataContractError(ctx.message(f"read_parquet failed: {type(exc).__name__}: {exc}")) from exc

    if df.empty:
        raise DataContractError(ctx.message("dataframe is empty"))
    if len(df.columns) == 0:
        raise DataContractError(ctx.message("dataframe has zero columns"))

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise DataContractError(ctx.message(f"missing required columns: {missing}"))

    return df
