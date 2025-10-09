import numpy as np
import matplotlib

matplotlib.use('Agg', force=True)


def test_summary_single_physics_panels():
    from mms_mp import visualize

    t = np.array(['2019-01-27T12:00:00', '2019-01-27T12:00:10', '2019-01-27T12:00:20'], dtype='datetime64[s]')
    B = np.column_stack([
        np.linspace(10.0, 12.0, t.size),
        np.linspace(-5.0, -4.0, t.size),
        np.linspace(1.0, 1.5, t.size),
    ])
    N_i = np.full(t.size, 8.0)
    N_e = np.full(t.size, 8.2)
    N_he = np.full(t.size, 0.6)
    vN_i = np.linspace(-200.0, -180.0, t.size)
    vN_e = np.linspace(-210.0, -190.0, t.size)
    vN_he = np.linspace(-150.0, -140.0, t.size)

    axes = visualize.summary_single(
        t=t,
        B_lmn=B,
        N_i=N_i,
        N_e=N_e,
        N_he=N_he,
        vN_i=vN_i,
        vN_e=vN_e,
        vN_he=vN_he,
        show=False,
    )

    dyn_ax = getattr(axes[3], '_dynamic_axis', None)
    assert dyn_ax is not None
    assert dyn_ax.get_ylabel() == 'Dynamic Pressure (nPa)'

    charge_ax = axes[4]
    assert charge_ax.get_ylabel() == 'ΔN (cm$^{-3}$)'
    he_ax = getattr(charge_ax, '_he_fraction_axis', None)
    assert he_ax is not None
    assert he_ax.get_ylabel() == 'He$^+$ Fraction'


def test_summary_single_dynamic_pressure_physics():
    from mms_mp import visualize

    t = np.array(['2019-01-27T12:00:00', '2019-01-27T12:00:05', '2019-01-27T12:00:10'], dtype='datetime64[s]')
    B = np.column_stack([
        np.linspace(20.0, 22.0, t.size),
        np.linspace(-8.0, -7.5, t.size),
        np.linspace(5.0, 4.5, t.size),
    ])
    N_i = np.array([9.0, 9.5, 10.0])
    N_e = N_i + 0.1
    N_he = np.array([0.8, 0.75, 0.7])
    vN_i = np.array([-150.0, -140.0, -130.0])
    vN_e = np.array([-160.0, -150.0, -140.0])
    vN_he = np.array([-100.0, -95.0, -90.0])

    axes = visualize.summary_single(
        t=t,
        B_lmn=B,
        N_i=N_i,
        N_e=N_e,
        N_he=N_he,
        vN_i=vN_i,
        vN_e=vN_e,
        vN_he=vN_he,
        show=False,
    )

    B_mag_expected = np.linalg.norm(B, axis=1)
    np.testing.assert_allclose(axes[3].lines[0].get_ydata(), B_mag_expected)

    mp_kg = 1.67262192369e-27
    cm3_to_m3 = 1e6
    km_to_m = 1e3
    ion_expected = N_i * cm3_to_m3 * mp_kg * (vN_i * km_to_m) ** 2 * 1e9
    he_expected = N_he * cm3_to_m3 * mp_kg * 4.0 * (vN_he * km_to_m) ** 2 * 1e9

    dyn_ax = getattr(axes[3], '_dynamic_axis')
    np.testing.assert_allclose(dyn_ax.lines[0].get_ydata(), ion_expected)
    np.testing.assert_allclose(dyn_ax.lines[1].get_ydata(), he_expected)

    charge_ax = axes[4]
    np.testing.assert_allclose(charge_ax.lines[0].get_ydata(), N_e - N_i)

    he_ax = getattr(charge_ax, '_he_fraction_axis')
    np.testing.assert_allclose(he_ax.lines[0].get_ydata(), N_he / N_i)
