from __future__ import annotations

from pathlib import Path

from scie_revision.common import scie_path


def generate_availability_statement(workbook_path: Path, output_dir: Path | None = None) -> Path:
    out = output_dir or scie_path("outputs", "scie_revision_round3", "reports")
    license_exists = any((Path(".") / name).exists() for name in ["LICENSE", "LICENSE.txt", "COPYING"])
    text = "\n".join(
        [
            "# Code and Data Availability Statement",
            "",
            "- Code can be shared: yes, this repository contains the full revision pipeline and manuscript-support scripts.",
            "- Dataset can be shared: available upon reasonable request. The upstream data provider is not fully documented in the workbook metadata and requires author confirmation.",
            "- An anonymized processed sample can be provided upon request for reproducibility.",
            f"- Reproducibility command: python run_scie_revision.py --data {workbook_path.as_posix()} --output outputs/scie_revision_round3 --seeds 11 17 23 --main_train_ratio 0.8 --horizon 3 --lookback 14 --max_tokens_for_modeling 200 --run_baselines --run_ablation --run_leakage_audit --run_output_quality_gate",
            "- Expected outputs directory: outputs/scie_revision_round3",
            "- Environment file path: requirements.txt",
            f"- License present: {'yes' if license_exists else 'no, consider adding one'}",
        ]
    )
    target = out / "code_data_availability_statement.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target
