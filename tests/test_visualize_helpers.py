import numpy as np
import matplotlib

matplotlib.use('Agg', force=True)

import matplotlib.pyplot as plt


def test_overlay_multi_renders_physics_time_alignment():
    from mms_mp import visualize

    dt = np.linspace(-5.0, 5.0, 5)
    overlay = {
        'Vn': {
            '1': np.column_stack([dt, np.linspace(-80.0, -60.0, dt.size)]),
            '2': np.column_stack([dt + 0.5, np.linspace(-78.0, -58.0, dt.size)]),
            '3': np.column_stack([dt - 1.0, np.linspace(-82.0, -62.0, dt.size)]),
        }
    }

    ax = visualize.overlay_multi(
        overlay,
        var='Vn',
        ref_probe='1',
        probes=('1', '2', '3'),
        ylabel='V$_N$ (km/s)',
        title='Normal velocity comparison',
        show=False,
    )

    try:
        assert ax.get_xlabel() == 'Δt relative to MMS1 (s)'
        assert ax.get_ylabel() == 'V$_N$ (km/s)'
        assert ax.get_title() == 'Normal velocity comparison'
        assert any(line.get_linewidth() > 2 for line in ax.lines)
        assert any(np.isclose(line.get_xdata(), 0.0).any() for line in ax.lines)
    finally:
        plt.close(ax.figure)


def test_plot_magnetic_field_utc_axis():
    from mms_mp import visualize

    t = np.array(['2019-01-27T12:00:00', '2019-01-27T12:00:05'], dtype='datetime64[s]')
    B = np.column_stack([
        np.full(t.size, 10.0),
        np.full(t.size, -5.0),
        np.linspace(1.0, 1.5, t.size),
    ])

    fig, axes = plt.subplots(4, 1)
    try:
        visualize.plot_magnetic_field(axes[:3], t, B)
        assert axes[2].get_xlabel() == 'Time (UTC)'
    finally:
        plt.close(fig)


def test_plot_displacement_physics_annotations():
    from mms_mp import visualize

    t = np.array(['2019-01-27T12:00:00', '2019-01-27T12:00:10'], dtype='datetime64[s]')
    disp = np.array([0.0, 150.0])
    sigma = np.array([10.0, 15.0])

    ax = visualize.plot_displacement(t, disp, sigma=sigma, show=False)
    try:
        assert ax.get_ylabel() == 'Δs (km)'
        legend = ax.get_legend()
        assert legend is not None
        assert '±1σ' in [text.get_text() for text in legend.get_texts()]
        assert ax.get_xlabel() == 'Time (UTC)'
    finally:
        plt.close(ax.figure)
