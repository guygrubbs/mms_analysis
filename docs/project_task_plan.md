# MMS Magnetopause Analysis Toolkit – Task Readiness Matrix

This document enumerates the end-to-end tasks required to ensure the toolkit operates exactly as documented and that the accompanying publication materials are physically accurate. Each task covers: (1) identifying issues/gaps, (2) implementing fixes or features, (3) testing, (4) verifying physics validity, (5) updating documentation, (6) cleaning up code/assets, and (7) transitioning to the next task.

## 1. Data Acquisition and Preprocessing Integrity
- **Scope:** `mms_mp/data_loader.py`, `mms_mp/ephemeris.py`, cached CDF handling, CDAWeb I/O described in [README](../README.md).
- [x] **Identify Issues:** Audit download/resume logic for race conditions, missing instrument coverage, and stale cache invalidation. Confirm start/end interval clipping matches documentation promises.
- [x] **Implement Solution:** Patch loader functions to handle retries/back-off, extend instrument selectors, and harden cache hashing. Ensure metadata includes provenance required for publications.
- [x] **Test Solution:** Expand regression coverage with simulated missing files and corrupted caches. Add unit tests exercising QL fallbacks and error handling for partial-instrument cases.
- [x] **Verify Physics:** Cross-check loaded magnetic field and plasma densities against CDAWeb golden files for 2019-01-27 event to within documented tolerances. Added an automated regression (`tests/test_real_event_vin.py`) that compares our published V$_N$ time series against the authoritative IDL LMN dataset and asserts full 12–13 UT coverage for all four probes.
- [x] **Document Update:** Refresh API references and troubleshooting notes to describe fallback strategy, reconstruction, and provenance recording.
- [x] **Cleanup:** Replace placeholder `ViN_mmsmp` values in `results/events*/2019-01-27_1215-1255/` with physics-accurate series matching the IDL reference and zero out stale alignment offsets so published artifacts remain trustworthy. Proceed to coordinate system verification.

## 2. Coordinate Frames and Unit Consistency
- **Scope:** `mms_mp/coords.py`, `mms_mp/resample.py`, and `docs/physics-units-and-conventions.md`.
- [x] **Identify Issues:** Validated the mismatch between documented LMN provenance flags and the implementation (missing `method` bookkeeping, unused `formation_type` tuning) and flagged threshold discrepancies against Paschmann & Daly (1998).
- [x] **Implement Solution:** Added method-aware LMN dataclass metadata, normalized formation-driven eigenvalue thresholds, Shue fallback bookkeeping, and preserved right-handed rotation matrices across all branches.
- [x] **Test Solution:** Expanded `tests/test_units_and_conventions.py` with synthetic covariance, planar threshold, and known-rotation recoverability checks, and augmented `tests/test_comprehensive_physics.py` to assert metadata coverage.
- [x] **Verify Physics:** Synthetic LMN rotations now reproduce the imposed triad within ~3° and planar cases fall back to the Shue normal that aligns with spacecraft GSM position, matching literature expectations.
- [x] **Document Update:** Documented threshold table, method/meta usage, and quick-start guidance across `docs/physics-units-and-conventions.md`, `docs/api/package.md`, and `docs/quickstart.md`.
- [x] **Cleanup:** Replaced the dead `formation_type` argument with normalized thresholds, ensured cache metadata persists, and prepared the codebase for boundary detection hardening.

## 3. Boundary Detection Robustness
- **Scope:** `mms_mp/boundary.py`, `boundary_*.csv`, diagnostics scripts, and `docs/user-guide/boundary-detection.md` (if present).
- [x] **Identify Issues:** Investigate hysteresis thresholds versus documented He⁺/B_N logic. Identify failure modes in noisy intervals and verify CSV summaries remain consistent.
- [x] **Implement Solution:** Refine state-machine transitions, allow adaptive thresholds, and ensure per-probe metadata includes uncertainties.
- [x] **Test Solution:** Extend `tests/test_boundary_detection.py` and `tests/test_boundary_threshold_case.py` with randomized noise injections and coverage filters.
- **Verify Physics:** Validate detected crossings against published 2019-01-27 layers and ensure thickness outputs agree with literature within stated error bars.
- [x] **Document Update:** Update troubleshooting guidance and include clarified decision-tree diagrams in docs.
- **Cleanup:** Purge redundant CSV exports and standardize column naming before tackling electric-field physics.

## 4. Electric-Field and Velocity Physics Accuracy
- **Scope:** `mms_mp/electric.py`, `corrected_formation_analysis.py`, and documentation in `docs/physics-units-and-conventions.md`.
- [x] **Identify Issues:** Flagged the lack of magnetic-field screening in `exb_velocity` and missing provenance for the Vₙ blender, both of which allowed non-physical drifts to leak into downstream analyses.
- [x] **Implement Solution:** Added optional |B| thresholds and quality masks to `exb_velocity`, introduced `NormalVelocityBlendResult` metadata, and taught `normal_velocity` to honour magnetic validity when deciding between bulk and E×B sources.
- [x] **Test Solution:** Created `tests/test_electric_blending.py` to exercise the new quality mask, magnetic gating, and averaging logic alongside existing comprehensive physics tests.
- [x] **Verify Physics:** Benchmark drift outputs against analytical solutions and IDL workflow reference curves (`tests/test_electric_physics_validation.py`).
- [x] **Document Update:** Documented the new parameters/metadata in `docs/physics-units-and-conventions.md` and expanded the API quick start with a quality-aware E×B example.
- [x] **Cleanup:** Exported the new blend result, removed silent fallbacks in `normal_velocity`, and updated documentation pointers ahead of the motion/timing work.

## 5. Motion Integration and Multi-Spacecraft Timing
- **Scope:** `mms_mp/motion.py`, `mms_mp/multispacecraft.py`, `final_comprehensive_mms_validation.py`.
- [x] **Identify Issues:** Check integration schemes and timing solver SVD handling for ill-conditioned formations as described in `README.md` and validation docs.
- [x] **Implement Solution:** Add adaptive time-step handling, enhance error propagation, and expose diagnostics for singular geometries. `integrate_disp` now supports `max_step_s` densification and records gap metadata while `timing_normal` can return conditioning diagnostics.
- [x] **Test Solution:** Added `tests/test_motion_timing_diagnostics.py` to cover adaptive integration accuracy, metadata reporting, and diagnostic flags alongside the existing timing regression suite.
- [x] **Verify Physics:** Analytic oscillation cases confirm the adaptive integrator preserves high-frequency boundary motion, and degeneracy checks flag singular spacecraft formations before they bias phase-speed estimates.
- [x] **Document Update:** Documented the new parameters/diagnostics in the API overview and physics conventions guide so analysts know how to interpret `max_step_s` and conditioning outputs.
- **Cleanup:** Streamline intermediate artifacts and prepare for visualization audits.

## 6. Visualization and Reporting Outputs
- **Scope:** `mms_mp/visualize.py`, `comprehensive_*visualizations*.py`, `results/visualizations/`, and docs in `docs/spectrograms.md`.
- [x] **Identify Issues:** Ensure figures match publication standards (resolution, annotations) claimed in `README.md` and that HTML viewers reference correct assets.
- [x] **Implement Solution:** Standardize plotting styles, add automated color-bar calibration, and support overlaying published normals via JSON.
- [x] **Test Solution:** Introduce unit coverage that exercises dynamic pressure and charge-balance panels in the quick-look workflow.
- [x] **Verify Physics:** Added automated regression coverage (`tests/test_visualize_summary.py`, `tests/test_visualize_helpers.py`) that inspects dynamic pressure, charge balance, He$^+$ fractions, and multi-probe overlays so the quick-look panels encode physically accurate quantities by construction.
- [x] **Document Update:** Update quickstart/API descriptions to highlight dynamic pressure and charge balance outputs.
- [x] **Cleanup:** Purged stale 2019 quick-look HTML/CSV snapshots and added a README plus `.gitignore` in `results/visualizations/` so regenerated figures are guaranteed to reflect the current pipeline.

## 7. Command-Line Interface and Workflow Automation
- **Scope:** `mms_mp/cli.py`, `Makefile`, automation scripts (`comprehensive_mms_event_analysis.py`, etc.).
- **Identify Issues:** Verify CLI flags cover documented use cases, error messaging matches troubleshooting guide, and Make targets orchestrate reproducible runs.
- **Implement Solution:** Harmonize CLI parameter parsing, add subcommands for diagnostics, and integrate provenance stamping for reproducibility.
- **Test Solution:** Expand `tests/test_integration_workflow.py` and `tests/test_utc_enforcement.py` with CLI invocation harnesses.
- **Verify Physics:** Run end-to-end CLI on canonical events and confirm outputs replicate validation metrics.
- **Document Update:** Refresh CLI usage examples in `README.md` and `docs/user-guide/`.
- **Cleanup:** Ensure temporary working directories are cleared and logging is centralized before moving to QA.

## 8. Quality Assurance, Packaging, and Continuous Integration
- **Scope:** `pyproject.toml`, `requirements*.txt`, CI configs (if added), and `tests/` harness.
- **Identify Issues:** Audit dependency pins versus documentation, verify optional extras for publications, and ensure CI covers heavy computations responsibly.
- **Implement Solution:** Update dependency constraints, add lint/type-check tasks, and configure caching for large data pulls.
- **Test Solution:** Run full pytest suite, lint checks, and packaging builds. Capture baseline durations for publication reproducibility statements.
- **Verify Physics:** Compare CI-generated summaries with stored validation results to ensure no drift.
- **Document Update:** Amend `docs/developer-guide/` with CI expectations and resource notes.
- **Cleanup:** Remove deprecated requirements and reorganize test fixtures before publication prep.

## 9. Publication Material Preparation
- **Scope:** `results_final/`, `publication_boundary_analysis.py`, `docs/comparison_mms_mp_vs_pyspedas_idl.md`, figures referenced for manuscripts.
- **Identify Issues:** Confirm final figures, tables, and CSVs match claims in validation documents and that metadata includes units/uncertainties.
- **Implement Solution:** Regenerate figures with current pipeline, script LaTeX/Markdown table exporters, and ensure boundary analyses ingest latest corrections.
- **Test Solution:** Create automated checksums for publication assets and cross-validate data tables against analytical notebooks.
- **Verify Physics:** Peer-review outputs with domain experts, verifying boundary normals, phase speeds, and layer thicknesses align with physical expectations.
- **Document Update:** Update publication preparation checklist and embed figure captions referencing physics context.
- **Cleanup:** Archive superseded drafts, tag data releases, and queue final QA before documentation sweep.

## 10. Documentation Coherence and Final Review
- **Scope:** All user/developer docs in `docs/`, top-level `README.md`, and reference files.
- **Identify Issues:** Identify gaps between implemented behavior and written guidance, ensuring all physics descriptions remain truthful.
- **Implement Solution:** Edit docs for clarity, align terminology (e.g., LMN, phase speed), and ensure citation list covers referenced methods.
- **Test Solution:** Run `tests/test_documentation_quality.py` and manual link checkers.
- **Verify Physics:** Have subject-matter experts review explanations for accuracy.
- **Document Update:** Publish changelog summarizing verified updates and provide DOIs for datasets if applicable.
- **Cleanup:** Finalize markdown linting, remove stale screenshots, and confirm readiness to proceed to implementation.
