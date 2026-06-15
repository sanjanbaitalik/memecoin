# Repository Inspection Summary

- `src/prism/pipeline.py`: existing end-to-end pipeline and old output wiring.
- `src/prism/data/workbook.py`: workbook discovery, auditing, and sheet loading.
- `src/prism/data/preprocess.py`: chronological preprocessing and lagged feature construction.
- `src/prism/baselines/runner.py`: original baseline experiment wrapper with fallback behavior.
- `src/prism/models/prism_variants.py`: existing PRISM ablation logic.
- `src/prism/stats/significance.py`: paired Wilcoxon helper used by legacy reports.
- `src/scie_revision/common.py`: shared output/table helpers for the new revision layer.
- `src/models/baselines/*.py`: trainable baseline wrappers for the revised comparison table.
- `src/metrics/*.py`: risk-adjusted and diversification metric helpers.
- `src/audits/*.py`: leakage and token-selection bias audits.
- `src/reporting/*.py`: claim framing, dataset card, MIS explanation, complexity, availability, and style reports.
- `run_scie_revision.py`: new master runner for the SCIE revision output package.
