"""Physics validation for the 2019-01-27 event using IDL gold-standard data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import readsav

IDL_SAV = Path('references/IDL_Code/mp_lmn_systems_20190127_1215_1255_mp_ver2.sav')
EVENT_DIRS = [
    Path('results/events_pub/2019-01-27_1215-1255'),
]


@pytest.fixture(scope='module')
def idl_vn() -> dict[str, tuple[pd.DatetimeIndex, np.ndarray]]:
    if not IDL_SAV.exists():
        pytest.skip('IDL reference file missing; cannot validate event physics')
    sav = readsav(IDL_SAV)
    out: dict[str, tuple[pd.DatetimeIndex, np.ndarray]] = {}
    for probe in ('1', '2', '3', '4'):
        rec = sav[f'vi_lmn{probe}'][0]
        times = pd.to_datetime(np.array(rec[0], dtype=np.float64), unit='s', utc=True)
        vn = np.array(rec[1][2], dtype=np.float64)
        out[probe] = (pd.DatetimeIndex(times), vn)
    return out


@pytest.mark.parametrize('directory', EVENT_DIRS)
@pytest.mark.parametrize('probe', ('1', '2', '3', '4'))
def test_vn_matches_idl_reference(directory: Path, probe: str, idl_vn):
    times_ref, vn_ref = idl_vn[probe]
    csv_path = directory / f'vn_probe{probe}.csv'
    assert csv_path.exists(), f'missing time series for probe {probe} in {directory}'
    df = pd.read_csv(csv_path)
    df['time_utc'] = pd.to_datetime(df['time_utc'], utc=True)
    assert not df['ViN_mmsmp'].isna().any(), 'mms_mp V_N contains NaNs'
    assert not df['ViN_sav'].isna().any(), 'IDL V_N column has NaNs'

    merged = pd.merge(
        df,
        pd.DataFrame({'time_utc': times_ref, 'Vn_idl': vn_ref}),
        on='time_utc',
        how='inner',
    )
    assert len(merged) == len(times_ref)

    diff_pipeline = np.abs(merged['ViN_mmsmp'] - merged['Vn_idl'])
    diff_idl = np.abs(merged['ViN_sav'] - merged['Vn_idl'])
    assert diff_idl.max() < 1e-9
    assert diff_pipeline.max() < 1e-2, 'Pipeline V_N deviates from IDL reference by more than 0.01 km/s'

    # Verify full 12–13 UT coverage with no gaps > 10 s
    window_mask = (merged['time_utc'] >= pd.Timestamp('2019-01-27T12:00:00Z')) & (
        merged['time_utc'] < pd.Timestamp('2019-01-27T13:00:00Z')
    )
    window = merged.loc[window_mask]
    assert len(window) > 0, 'No samples found between 12–13 UT'
    expected_count = int(((times_ref >= pd.Timestamp('2019-01-27T12:00:00Z')) &
                          (times_ref < pd.Timestamp('2019-01-27T13:00:00Z'))).sum())
    assert len(window) == expected_count
    diffs = np.diff(window['time_utc'].values.astype('datetime64[ns]').astype('int64')) / 1e9
    assert diffs.max() <= 10.0, 'Gaps larger than 10 s detected within the hour'
    assert np.allclose(merged['dt_ms'], 0.0), 'Alignment offsets should be zero after reprocessing'
