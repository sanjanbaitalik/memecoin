from __future__ import annotations

from pathlib import Path

from scie_revision.common import scie_path


def generate_availability_statement(workbook_path: Path, output_dir: Path | None = None) -> Path:
    out = output_dir or scie_path("outputs", "scie_revision", "reports")
    license_exists = any((scie_path(".") / name).exists() for name in ["LICENSE", "LICENSE.txt", "COPYING"])
    text = "\n".join(
        [
            "# Code and Data Availability Statement",
            "",
            "- Code can be shared: yes, this repository contains the full revision pipeline and manuscript-support scripts.",
            "- Dataset can be shared: not fully public from the workbook metadata alone.",
            "- Dataset limitation: the upstream provider details are not fully documented in the workbook metadata, so the repository only asserts the local workbook file used for processing.",
            f"- Reproducibility command: python run_scie_revision.py --data {workbook_path.as_posix()} --output outputs/scie_revision --seeds 11 17 23 --main_train_ratio 0.8 --horizon 3 --lookback 14 --run_baselines --run_prism --run_ablation --run_robustness --run_risk_metrics --run_audits",
            "- Expected outputs directory: outputs/scie_revision",
            "- Environment file path: requirements.txt",
            f"- License present: {'yes' if license_exists else 'no, consider adding one'}",
        ]
    )
    target = out / "code_data_availability_statement.md"
    target.write_text(text, encoding="utf-8")
    return target
