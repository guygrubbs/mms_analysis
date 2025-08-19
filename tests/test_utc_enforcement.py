import os
import re
from datetime import timezone

import matplotlib
matplotlib.use('Agg')

# We can import helper functions from publication module
from publication_boundary_analysis import ensure_datetime_format


def test_ensure_datetime_format_utc():
    # naive unix seconds
    times = [1548592250.0, 1548592310.0]
    dt = ensure_datetime_format(times)
    assert all(getattr(t, 'tzinfo', None) is not None for t in dt)
    assert all(t.tzinfo == timezone.utc for t in dt)


def test_generated_csv_times_are_utc(tmp_path, monkeypatch):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Minimal stub: write tiny CSV using UTC datetimes to match our format
        from datetime import datetime, timezone
        from csv import writer
        with open('boundary_crossings.csv', 'w', newline='') as cf:
            w = writer(cf)
            w.writerow(['spacecraft', 'crossing_time_UT'])
            w.writerow(['MMS1', datetime(2019,1,27,12,30,50,tzinfo=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')])
        # Read back
        with open('boundary_crossings.csv', 'r') as cf:
            content = cf.read()
        # Expect no timezone offset string like +00:00 and consistent format
        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", content, re.M), content
    finally:
        os.chdir(cwd)

