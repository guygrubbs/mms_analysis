Archive structure

This directory contains files and directories that were archived to keep the repository focused on current, validated results and code.

Subdirectories
- old_results/: Outdated or superseded figures and CSVs from earlier exploratory runs (e.g., boundary_* artifacts and top‑level MMS figures from 2019‑01‑27), plus archived pre‑validation outputs for the 2019‑01‑27 event (e.g., `old_results/2019-01-27_pre-validation_20251117/`). These are retained for provenance but are not used by tests or current analysis.
- old_code/: Scratch or debugging scripts that are not part of the current workflow.

Notes
- Canonical outputs for the validated event live in results/events_pub/2019-01-27_1215-1255/ and are reproducible by running:
  - py -3.11 examples/analyze_20190127_dn_shear.py
- Strict local caching is enforced in the analysis; no data are re‑downloaded if they exist locally.
- If you need to restore anything for historical comparison, move files back explicitly and open a PR describing the reason.

