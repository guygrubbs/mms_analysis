import numpy as np
from pyspedas import mms
from pytplot import get_data
import matplotlib.pyplot as plt

def try_get(name):
    """Return (t, y) if a tplot var exists, else (None, None)."""
    out = get_data(name)
    return out if out is not None else (None, None)

# ---------------------------------------------------------------------------
# revised loader ------------------------------------------------------------
# ---------------------------------------------------------------------------
def load_mms_data(trange):
    """
    Download + return MMS MEC, FGM and FPI moment data.
    Electron bulk V is optional; code keeps running if it’s absent.
    """
    # ---- download (kept in tplot) -----------------------------------------
    mms.mec(trange=trange, probe=['1','2','3','4'],
            data_rate='srvy', level='l2', notplot=False)          # pos
    mms.fgm(trange=trange, probe=['1','2','3','4'],
            data_rate='srvy', level='l2', notplot=False)          # B
    mms.fpi(trange=trange, probe=['1','2','3','4'],
            data_rate='fast', level='l2',
            datatype=['dis-moms', 'des-moms'], notplot=False)     # moments

    # ---- collect into python dict ----------------------------------------
    data = {}
    for sc in ['1','2','3','4']:
        sid = f"mms{sc}"

        # MEC position (GSE, km)
        t_pos, pos = try_get(f"{sid}_mec_r_gse")
        if t_pos is None:
            raise RuntimeError(f"{sid} position var missing – MEC failed to load")

        # FGM magnetic field (GSE, nT)
        t_b, B = try_get(f"{sid}_fgm_b_gse_srvy_l2")
        if t_b is None:
            raise RuntimeError(f"{sid} B var missing – FGM failed to load")

        # Ion bulk V (GSE, km/s) – always present in fast moments
        t_vi, Vi = try_get(f"{sid}_dis_bulkv_gse_fast")
        if t_vi is None:
            raise RuntimeError(f"{sid} ion bulk-V missing – DIS moments failed")

        # Electron bulk V – *may* be absent in fast L2; keep going if so
        t_ve, Ve = try_get(f"{sid}_des_bulkv_gse_fast")   # may return (None, None)

        # Densities (optional, for edge picking)
        _, Ni = try_get(f"{sid}_dis_numberdensity_fast")
        _, Ne = try_get(f"{sid}_des_numberdensity_fast")

        data[sid] = {
            'time_pos': t_pos, 'pos': pos,
            'time_b': t_b,     'B': B,
            'time_vi': t_vi,   'Vi': Vi,
            'time_ve': t_ve,   'Ve': Ve,   # can be None
            'Ni': Ni, 'Ne': Ne
        }
    return data


def confirm_string_of_pearls(mms_data):
    # Check radial distances and separations for MMS1-4
    ref_time = (mms_data['mms1']['time_pos'][0] + mms_data['mms1']['time_pos'][-1]) / 2
    ref_positions = {}
    for sc_id in ['mms1','mms2','mms3','mms4']:
        t = mms_data[sc_id]['time_pos']; pos = mms_data[sc_id]['pos']
        ref_positions[sc_id] = np.array([np.interp(ref_time, t, pos[:,0]),
                                         np.interp(ref_time, t, pos[:,1]),
                                         np.interp(ref_time, t, pos[:,2])])
    rad_dists = {sc: np.linalg.norm(vec)/6371.0 for sc, vec in ref_positions.items()}
    sep12 = np.linalg.norm(ref_positions['mms1'] - ref_positions['mms2'])
    sep23 = np.linalg.norm(ref_positions['mms2'] - ref_positions['mms3'])
    sep34 = np.linalg.norm(ref_positions['mms3'] - ref_positions['mms4'])
    print("Radial distance at midpoint (Re):")
    for sc, r in rad_dists.items():
        print(f"  {sc.upper()}: {r:.2f} Re")
    print("Neighbor separations at midpoint (km):")
    print(f"  MMS1-2: {sep12:.1f} km, MMS2-3: {sep23:.1f} km, MMS3-4: {sep34:.1f} km")

def compute_normal_velocity(mms_data, normal_vector):
    N = np.array(normal_vector, dtype=float); N = N/np.linalg.norm(N)
    # Pick an arbitrary M perpendicular to N (cross with Z or X)
    z_hat = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(N, z_hat)) < 0.99:
        M = np.cross(N, z_hat)
    else:
        M = np.cross(N, np.array([1.0, 0.0, 0.0]))
    M = M/np.linalg.norm(M)
    L = np.cross(M, N); L = L/np.linalg.norm(L)
    Vn_data = {}
    for sc_id in ['mms1','mms2','mms3','mms4']:
        vi = mms_data[sc_id]['Vi']; t_vi = mms_data[sc_id]['time_vi']
        Vn = np.dot(vi, N)
        Vn_data[sc_id] = {'time': t_vi, 'Vn': Vn}
    return Vn_data

def integrate_boundary_motion(time, Vn, t_start, t_end):
    # Integrate Vn from t_start to t_end
    if isinstance(time[0], np.datetime64):
        time_num = time.astype('datetime64[ns]').astype('int64') * 1e-9
    else:
        time_num = time
    mask = (time_num >= (np.datetime64(t_start).astype('datetime64[ns]').astype('int64')*1e-9)) & \
           (time_num <= (np.datetime64(t_end).astype('datetime64[ns]').astype('int64')*1e-9))
    if not np.any(mask):
        return 0.0
    t_seg = time_num[mask].astype(float)
    Vn_seg = Vn[mask]
    dt = np.diff(t_seg)
    displacement = np.sum((Vn_seg[:-1] + Vn_seg[1:]) / 2 * dt)  # km
    return displacement

def compute_distance_time_series(time, Vn, ref_time):
    # Compute continuous distance profile, with 0 at ref_time
    if isinstance(time[0], np.datetime64):
        t_num = time.astype('datetime64[ns]').astype('int64') * 1e-9
    else:
        t_num = time.astype(float)
    t0 = float(t_num[0])
    t_sec = t_num - t0
    # Cumulative integral of Vn
    dist = np.concatenate(([0.0], np.cumsum((Vn[:-1] + Vn[1:]) / 2 * np.diff(t_sec))))
    # Reference distance at ref_time
    ref_sec = (np.datetime64(ref_time).astype('datetime64[ns]').astype('int64') * 1e-9) - t0
    ref_dist = np.interp(ref_sec, t_sec, dist)
    return dist - ref_dist

def plot_distance_time_series(distance_series, boundary_times):
    plt.figure(figsize=(10,6))
    colors = {'mms1':'tab:red', 'mms2':'tab:blue', 'mms3':'tab:green', 'mms4':'tab:purple'}
    markers = {'ion_edge': '^', 'curr_sheet': 'x', 'electron_edge': 'v'}
    for sc_id, data in distance_series.items():
        t = data['time']; dist = data['distance']; c = colors[sc_id]
        plt.plot(t, dist, label=sc_id.upper(), color=c)
        # Mark ion edge, current sheet, electron edge
        for edge, m in markers.items():
            t_edge = boundary_times[sc_id][edge]
            # interpolate distance at t_edge:
            t_edge_num = np.datetime64(t_edge).astype('datetime64[ns]').astype('int64') * 1e-9
            t_num = t.astype('datetime64[ns]').astype('int64') * 1e-9
            d_edge = np.interp(t_edge_num, t_num, dist)
            plt.scatter(t_edge, d_edge, marker=m, color=c, edgecolors='k', zorder=5)
    plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
    plt.title("MMS Magnetopause Distance (2019-01-27 12:00-13:00 UT)")
    plt.xlabel("Time (UT)")
    plt.ylabel("Distance to Magnetopause (km)")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.savefig("MMS_magnetopause_distance_20190127.png", dpi=150)
    plt.show()

# Main execution
trange = ['2019-01-27/12:00:00', '2019-01-27/13:00:00']
mms_data = load_mms_data(trange)
confirm_string_of_pearls(mms_data)
# Define normal (LMN) - given
normal_vector = [0.98, -0.05, 0.18]
Vn_data = compute_normal_velocity(mms_data, normal_vector)
# Define boundary crossing times (manually identified from data)
boundary_times = {
    'mms1': {'ion_edge': np.datetime64('2019-01-27T12:23:05'), 'curr_sheet': np.datetime64('2019-01-27T12:23:12'), 'electron_edge': np.datetime64('2019-01-27T12:23:18')},
    'mms2': {'ion_edge': np.datetime64('2019-01-27T12:24:15'), 'curr_sheet': np.datetime64('2019-01-27T12:24:22'), 'electron_edge': np.datetime64('2019-01-27T12:24:29')},
    'mms3': {'ion_edge': np.datetime64('2019-01-27T12:25:25'), 'curr_sheet': np.datetime64('2019-01-27T12:25:33'), 'electron_edge': np.datetime64('2019-01-27T12:25:40')},
    'mms4': {'ion_edge': np.datetime64('2019-01-27T12:26:35'), 'curr_sheet': np.datetime64('2019-01-27T12:26:43'), 'electron_edge': np.datetime64('2019-01-27T12:26:50')}
}
# Compute boundary thickness via Vn integration for each spacecraft
for sc_id in ['mms1','mms2','mms3','mms4']:
    t_arr = Vn_data[sc_id]['time']; vn_arr = Vn_data[sc_id]['Vn']
    t_i = boundary_times[sc_id]['ion_edge']; t_e = boundary_times[sc_id]['electron_edge']
    disp = integrate_boundary_motion(t_arr, vn_arr, t_i, t_e)
    print(f"{sc_id.upper()} boundary layer thickness ~ {disp:.0f} km")
# Estimate global normal velocity from MMS1 vs MMS4 timing
t1 = boundary_times['mms1']['curr_sheet']; t4 = boundary_times['mms4']['curr_sheet']
pos1 = mms_data['mms1']['pos'][-1]  # using last position as approx at crossing (or interpolate)
pos4 = mms_data['mms4']['pos'][-1]
N = np.array(normal_vector); N = N/np.linalg.norm(N)
delta_d = np.dot((pos4 - pos1), N)
dt = (t4 - t1) / np.timedelta64(1, 's')
Vn_global = delta_d / dt
print(f"Global boundary normal speed ~ {Vn_global:.1f} km/s based on MMS1-4 timing")
# Predict crossing for MMS2, MMS3
for sc_id in ['mms2','mms3']:
    pos_sc = mms_data[sc_id]['pos'][-1]
    delta_d_sc = np.dot((pos_sc - pos1), N)
    t_pred = t1 + np.timedelta64(int(delta_d_sc / Vn_global * 1000), 'ms')
    t_obs = boundary_times[sc_id]['curr_sheet']
    print(f"{sc_id.upper()} predicted CS time ~ {t_pred} vs observed ~ {t_obs}")
# Compute and plot distance profiles
distance_series = {}
for sc_id in ['mms1','mms2','mms3','mms4']:
    distance_series[sc_id] = {
        'time': Vn_data[sc_id]['time'],
        'distance': compute_distance_time_series(Vn_data[sc_id]['time'], Vn_data[sc_id]['Vn'], boundary_times[sc_id]['curr_sheet'])
    }
plot_distance_time_series(distance_series, boundary_times)
