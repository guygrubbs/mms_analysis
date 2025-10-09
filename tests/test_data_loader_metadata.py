import numpy as np
import pytest
from unittest.mock import Mock, patch


@patch('mms_mp.data_loader.get_data')
@patch('mms_mp.data_loader.mms')
def test_load_event_provenance(mock_mms, mock_get_data, monkeypatch):
    from mms_mp import data_loader

    # Prepare synthetic time series aligned with requested window
    base = np.datetime64('2019-01-27T12:00:00')
    times = base + np.arange(5) * np.timedelta64(10, 's')

    vector = np.column_stack([
        np.full(5, 10.0),
        np.full(5, -5.0),
        np.linspace(1.0, 3.0, 5),
    ])
    scalar = np.linspace(5.0, 9.0, 5)
    zeros = np.zeros(5)

    data_map = {
        'mms1_dis_numberdensity_fast': (times, scalar),
        'mms1_dis_bulkv_gse_fast': (times, vector),
        'mms1_fgm_b_gsm_fast': (times, vector),
        'mms1_des_numberdensity_fast': (times, scalar * 0.8),
        'mms1_des_bulkv_gse_fast': (times, vector * 0.5),
        'mms1_hpca_heplus_number_density_moments': (times, scalar * 0.02),
        'mms1_hpca_heplus_ion_bulk_velocity_moments': (times, vector * 0.1),
        'mms1_edp_dce_fast_l2': (times, vector * 0.01),
        'mms1_edp_scpot_fast_l2': (times, scalar * 0.0),
        'mms1_mec_r_gsm': (times, np.full((5, 3), 64000.0)),
        'mms1_mec_v_gsm': (times, np.full((5, 3), 1.0)),
        'mms1_dis_quality_flag_fast': (times, zeros),
        'mms1_des_quality_flag_fast': (times, zeros + 1),
        'mms1_hpca_status_flag_fast': (times, zeros),
    }

    mock_get_data.side_effect = lambda name: data_map[name]
    monkeypatch.setattr(data_loader, '_tp', mock_get_data, raising=False)

    dq = {
        'mms1_dis_numberdensity_fast': None,
        'mms1_dis_bulkv_gse_fast': None,
        'mms1_fgm_b_gsm_fast': None,
        'mms1_des_numberdensity_fast': None,
        'mms1_des_bulkv_gse_fast': None,
        'mms1_hpca_heplus_number_density_moments': None,
        'mms1_hpca_heplus_ion_bulk_velocity_moments': None,
        'mms1_edp_dce_fast_l2': None,
        'mms1_edp_scpot_fast_l2': None,
        'mms1_mec_r_gsm': None,
        'mms1_mec_v_gsm': None,
        'mms1_dis_quality_flag_fast': None,
        'mms1_des_quality_flag_fast': None,
        'mms1_hpca_status_flag_fast': None,
    }
    monkeypatch.setattr(data_loader, 'data_quants', dq, raising=False)

    # Ensure loader entry points exist
    mock_mms.mms_load_fgm = Mock()
    mock_mms.mms_load_fpi = Mock()
    mock_mms.mms_load_hpca = Mock()
    mock_mms.mms_load_edp = Mock()
    mock_mms.mms_load_mec = Mock()
    mock_mms.mms_load_state = Mock()

    # Ensure internal shortcuts use the patched getter and MEC cache is populated
    monkeypatch.setattr(data_loader, '_tp', mock_get_data, raising=False)

    def _fake_load_state(trange, probe, *, download_only=False, mec_cache=None):
        if mec_cache is not None:
            mec_cache[str(probe)] = {
                'pos': data_map['mms1_mec_r_gsm'],
                'vel': data_map['mms1_mec_v_gsm'],
                'pos_name': 'mms1_mec_r_gsm',
                'vel_name': 'mms1_mec_v_gsm',
                'source': 'mec',
            }
        return None

    monkeypatch.setattr(data_loader, '_load_state', _fake_load_state, raising=False)

    evt = data_loader.load_event(
        ['2019-01-27T12:00:00', '2019-01-27T12:00:40'],
        probes=['1'],
        include_hpca=True,
        include_edp=True,
        include_ephem=True,
    )

    assert '1' in evt
    meta = evt['__meta__']
    assert meta['probes'] == ['1']
    assert meta['ephemeris_sources']['1'] == 'mec'
    assert meta['download_summary']['fgm']['success'] is True
    assert meta['sources']['1']['B_gsm'] == 'mms1_fgm_b_gsm_fast'
    assert meta['time_coverage']['1']['B_gsm']['start'].startswith('2019-01-27T12:00:00')
    assert meta['time_coverage']['1']['B_gsm']['end'].startswith('2019-01-27T12:00:40')
    assert meta['warnings'] == []

    # Quality flags should be included under canonical keys
    assert f"mms1_dis_quality_flag" in evt['1']
    assert f"mms1_des_quality_flag" in evt['1']
    assert f"mms1_hpca_status_flag" in evt['1']

    # Ensure placeholders were not used for main variables
    b_times, b_data = evt['1']['B_gsm']
    assert b_data.shape == (5, 3)
    assert np.isfinite(b_data).all()


@patch('mms_mp.data_loader.get_data')
@patch('mms_mp.data_loader.mms')
def test_load_event_recovers_ql_moments(mock_mms, mock_get_data, monkeypatch):
    from mms_mp import data_loader

    base = np.datetime64('2019-01-27T12:00:00')
    times = base + np.arange(5) * np.timedelta64(10, 's')
    vector = np.column_stack([
        np.linspace(10.0, 12.0, 5),
        np.linspace(-4.0, -6.0, 5),
        np.linspace(1.0, 2.5, 5),
    ])
    scalar = np.linspace(6.0, 9.0, 5)

    data_map = {
        'mms1_fgm_b_gsm_fast': (times, vector),
        'mms1_mec_r_gsm': (times, np.full((5, 3), 64000.0)),
        'mms1_mec_v_gsm': (times, np.full((5, 3), 1.0)),
    }

    dq = {
        'mms1_fgm_b_gsm_fast': None,
        'mms1_mec_r_gsm': None,
        'mms1_mec_v_gsm': None,
    }

    monkeypatch.setattr(data_loader, 'data_quants', dq, raising=False)

    mock_get_data.side_effect = lambda name: data_map[name]
    monkeypatch.setattr(data_loader, '_tp', mock_get_data, raising=False)

    def _fake_load(instr, rates, *, trange, probe, download_only=False, status=None, **extra):
        level = extra.get('level')
        datatype = extra.get('datatype')
        if status is not None:
            status.update({'rates_requested': rates, 'success': True, 'used_rate': rates[0], 'attempts': []})
        if instr == 'fpi' and level == 'ql' and datatype == 'dis-moms':
            data_map['mms1_dis_numberdensity_ql'] = (times, scalar * 1.1)
            data_map['mms1_dis_bulkv_gse_ql'] = (times, vector * 0.8)
            dq['mms1_dis_numberdensity_ql'] = None
            dq['mms1_dis_bulkv_gse_ql'] = None
        if instr == 'fpi' and level == 'ql' and datatype == 'des-moms':
            data_map['mms1_des_numberdensity_ql'] = (times, scalar * 0.9)
            data_map['mms1_des_bulkv_gse_ql'] = (times, vector * 0.6)
            dq['mms1_des_numberdensity_ql'] = None
            dq['mms1_des_bulkv_gse_ql'] = None
        return rates[0]

    monkeypatch.setattr(data_loader, '_load', _fake_load, raising=False)

    def _fake_load_state(trange, probe, *, download_only=False, mec_cache=None):
        if mec_cache is not None:
            mec_cache[str(probe)] = {
                'pos': data_map['mms1_mec_r_gsm'],
                'vel': data_map['mms1_mec_v_gsm'],
                'pos_name': 'mms1_mec_r_gsm',
                'vel_name': 'mms1_mec_v_gsm',
                'source': 'mec',
            }
        return None

    monkeypatch.setattr(data_loader, '_load_state', _fake_load_state, raising=False)

    evt = data_loader.load_event(
        ['2019-01-27T12:00:00', '2019-01-27T12:00:40'],
        probes=['1'],
        include_hpca=False,
        include_edp=False,
        include_ephem=True,
    )

    meta = evt['__meta__']
    assert meta['download_summary']['fpi_dis_ql']['success'] is True
    assert meta['download_summary']['fpi_des_ql']['success'] is True
    assert meta['sources']['1']['N_tot'] == 'mms1_dis_numberdensity_ql'
    assert meta['sources']['1']['N_e'] == 'mms1_des_numberdensity_ql'
    assert np.isfinite(evt['1']['N_tot'][1]).all()
    assert np.isfinite(evt['1']['N_e'][1]).all()


@patch('mms_mp.data_loader.get_data')
@patch('mms_mp.data_loader.mms')
def test_load_event_errors_when_ion_density_missing(mock_mms, mock_get_data, monkeypatch):
    from mms_mp import data_loader

    base = np.datetime64('2019-01-27T12:00:00')
    times = base + np.arange(5) * np.timedelta64(10, 's')
    vector = np.column_stack([
        np.linspace(10.0, 12.0, 5),
        np.linspace(-4.0, -6.0, 5),
        np.linspace(1.0, 2.5, 5),
    ])

    data_map = {
        'mms1_fgm_b_gsm_fast': (times, vector),
        'mms1_mec_r_gsm': (times, np.full((5, 3), 64000.0)),
        'mms1_mec_v_gsm': (times, np.full((5, 3), 1.0)),
    }

    dq = {
        'mms1_fgm_b_gsm_fast': None,
        'mms1_mec_r_gsm': None,
        'mms1_mec_v_gsm': None,
    }

    monkeypatch.setattr(data_loader, 'data_quants', dq, raising=False)
    mock_get_data.side_effect = lambda name: data_map[name]
    monkeypatch.setattr(data_loader, '_tp', mock_get_data, raising=False)

    def _fake_load(instr, rates, *, trange, probe, download_only=False, status=None, **extra):
        if status is not None:
            status.update({'rates_requested': rates, 'success': True, 'used_rate': None, 'attempts': []})
        return None

    monkeypatch.setattr(data_loader, '_load', _fake_load, raising=False)

    def _fake_load_state(trange, probe, *, download_only=False, mec_cache=None):
        if mec_cache is not None:
            mec_cache[str(probe)] = {
                'pos': data_map['mms1_mec_r_gsm'],
                'vel': data_map['mms1_mec_v_gsm'],
                'pos_name': 'mms1_mec_r_gsm',
                'vel_name': 'mms1_mec_v_gsm',
                'source': 'mec',
            }
        return None

    monkeypatch.setattr(data_loader, '_load_state', _fake_load_state, raising=False)

    with pytest.raises(RuntimeError, match='N_tot'):
        data_loader.load_event(
            ['2019-01-27T12:00:00', '2019-01-27T12:00:40'],
            probes=['1'],
            include_hpca=False,
            include_edp=False,
            include_ephem=True,
        )
