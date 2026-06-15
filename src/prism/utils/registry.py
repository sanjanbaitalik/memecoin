from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prism.utils.paths import output_path


def write_run_manifest(manifest: dict[str, Any]) -> Path:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        **manifest,
    }
    target = output_path("outputs", "logs", "run_manifest.json")
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target
