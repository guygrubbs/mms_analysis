# UTC Normalization and Spectrogram Diagnostics

This repository enforces UTC for all figures and analysis outputs and provides a diagnostics CSV for spectrogram variable resolution.

## UTC Normalization

- All date-time values plotted are timezone-aware UTC datetimes.
- Matplotlib is configured to format in UTC via `plt.rcParams['timezone'] = 'UTC'`.
- Utility: `ensure_datetime_format(times)` converts arrays of numeric or datetime-like values into Python datetimes with `tzinfo=UTC`.
- Example usage:

```
from publication_boundary_analysis import ensure_datetime_format
plot_times = ensure_datetime_format(times)
```

- CSV/JSON outputs store UT time strings using `strftime('%Y-%m-%d %H:%M:%S')`.

## Spectrogram Diagnostics

- During `test_boundary_threshold_case.py`, the code writes `spectrogram_diagnostics.csv` summarizing which tplot variable names were found or imputed per MMS and species.
- Columns:
  - `species`, `probe`, `candidate_or_dist_var`, `energy_var`, `energy_info`, `status`
- Status values:
  - `NOT_FOUND`: omni spectrogram variable not available
  - `MISSING_DIST_OR_ENERGY`: distribution or energy array missing for fallback build
  - `OMNI_NO_ENERGY_VECTOR`: omni spectrogram lacked an energy vector; bins imputed
  - `IMPUTED_ENERGY_VECTOR`: energy axis constructed (e.g., 32 log bins)

## CLI Controls

In `test_boundary_threshold_case.py`:
- `--ion-emax` and `--electron-emax` control low-energy caps for the focused spectrograms.
- `--include-grids/--no-grids` include per-MMS grid pages in the PDF.
- `--include-combined/--no-combined` include the 2×3 combined figure page.

## Running tests

Install dev dependencies, then run:

```
pytest -q
```

Relevant tests:
- `tests/test_utc_enforcement.py` verifies UTC formatting normalization and CSV time format contract.
- `tests/test_spectrogram_diagnostics.py` verifies diagnostics CSV schema and basic contents.

