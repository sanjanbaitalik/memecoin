from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prism.pipeline import run_full_pipeline  # noqa: E402
from prism.utils.paths import find_project_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PRISM revision pipeline end-to-end")
    parser.add_argument("--input", required=True, help="Path to workbook file")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root = find_project_root()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (root / input_path).resolve()

    if not input_path.exists():
        print(f"ERROR: input workbook does not exist: {input_path}")
        return 2

    os.environ["PRISM_INPUT_WORKBOOK"] = str(input_path)
    os.environ["PRISM_OUTDIR"] = args.outdir

    payload = run_full_pipeline()

    failed_steps = [s for s in payload.get("step_status", []) if s.get("status") == "failed"]
    print(json.dumps(payload.get("steps", {}), indent=2))
    print(f"Output directory: {(root / args.outdir).resolve()}")

    if failed_steps:
        print("Failed steps:")
        for step in failed_steps:
            print(f"- {step.get('step')}: {step.get('detail')}")
        return 1

    print("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
