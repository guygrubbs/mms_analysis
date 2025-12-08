import pandas as pd
from pathlib import Path
import pytest

EVENT_DIR = Path('results/events_pub/2019-01-27_1215-1255')


def test_output_files_exist_and_nonempty():
    expected = [
        'crossings_all_1243.csv',
        'crossings_mixed_1230_1243.csv',
        'predictions_all_1243.csv',
        'predictions_mixed_1230_1243.csv',
        'shear_all_1243.csv',
        'shear_mixed_1230_1243.csv',
        'summary_metrics.csv',
        'physical_interpretation.md',
    ]
    for name in expected:
        p = EVENT_DIR / name
        assert p.exists(), f"Missing {name}"
        assert p.stat().st_size > 0, f"{name} is empty"


def test_crossing_times_within_window():
    t0 = pd.to_datetime('2019-01-27T12:15:00', utc=True)
    t1 = pd.to_datetime('2019-01-27T12:55:00', utc=True)
    for name in ['crossings_all_1243.csv', 'crossings_mixed_1230_1243.csv']:
        df = pd.read_csv(EVENT_DIR / name)
        df['time_utc'] = pd.to_datetime(df['time_utc'], utc=True, errors='coerce')
        assert df['time_utc'].notna().all(), f"{name} has unparsable times"
        assert (df['time_utc'] >= t0).all(), f"{name} has times before window"
        assert (df['time_utc'] <= t1).all(), f"{name} has times after window"


@pytest.mark.parametrize('probe', ['1','2','3','4'])
@pytest.mark.parametrize('label', ['all_1243','mixed_1230_1243'])
def test_dn_physically_reasonable(probe, label):
    p = EVENT_DIR / f'dn_mms{probe}_{label}.csv'
    assert p.exists(), f"Missing DN file {p}"
    df = pd.read_csv(p)
    assert 'DN_km' in df.columns, f"DN_km column missing in {p.name}"
    # DN magnitudes for this event should be within +/- 1000 km
    max_abs = pd.to_numeric(df['DN_km'], errors='coerce').abs().max()
    assert max_abs < 1000, f"Unreasonable DN magnitude {max_abs} in {p.name}"


def test_shear_angles_within_bounds():
    for name in ['shear_all_1243.csv', 'shear_mixed_1230_1243.csv']:
        df = pd.read_csv(EVENT_DIR / name)
        assert 'shear_deg' in df.columns
        vals = pd.to_numeric(df['shear_deg'], errors='coerce')
        assert (vals >= 0).all(), f"{name} has negative shear"
        assert (vals <= 180).all(), f"{name} has shear > 180 deg"




def test_spectrogram_pngs_exist_and_nonempty():
    for probe in ['1', '2', '3', '4']:
        for tag in ['DIS', 'DES']:
            name = f'mms{probe}_{tag}_omni.png'
            p = EVENT_DIR / name
            assert p.exists(), f"Missing spectrogram {name}"
            assert p.stat().st_size > 0, f"Spectrogram {name} is empty"
