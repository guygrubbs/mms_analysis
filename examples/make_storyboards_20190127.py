"""
Create per-probe storyboard figures for publication:
- Row 1: DIS spectrogram
- Row 2: ViN overlay (sav vs mms_mp) with vt shading and MAE annotation
- Row 3: DN bar chart by vt segment
Saves to results/events_pub/2019-01-27_1215-1255/storyboard_mms{p}.png
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

EVENT_DIR = pathlib.Path('results/events_pub/2019-01-27_1215-1255')


def _load_csv(p: str) -> pd.DataFrame | None:
    f = EVENT_DIR / f'vn_probe{p}.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df['time_utc'] = pd.to_datetime(df['time_utc'], utc=True)
    df = df.set_index('time_utc')
    return df


def _load_dn_csv(p: str) -> pd.DataFrame | None:
    f = EVENT_DIR / f'mms{p}_DN_segments.csv'
    if f.exists():
        return pd.read_csv(f)
    return None


def _load_spectrogram_images(p: str) -> tuple[Image.Image | None, Image.Image | None]:
    dis = EVENT_DIR / f'mms{p}_DIS_omni.png'
    des = EVENT_DIR / f'mms{p}_DES_omni.png'
    img_dis = Image.open(dis) if dis.exists() else None
    img_des = Image.open(des) if des.exists() else None
    return img_dis, img_des


def _draw_vin_overlay(ax, df):
    ax.plot(df.index, df['ViN_sav'], label='IDL .sav ViN', lw=1.2)
    ax.plot(df.index, df['ViN_mmsmp'], label='mms_mp ViN', lw=1.0, alpha=0.9)
    ax.set_ylabel('V_N (km/s)')
    ax.grid(True, alpha=0.3)
    diff = (df['ViN_mmsmp'] - df['ViN_sav']).dropna()
    mae = float(np.nanmean(np.abs(diff))) if len(diff) else np.nan
    ax.text(0.01, 0.95, f"MAE = {mae:.2f} km/s", transform=ax.transAxes, va='top', ha='left')
    ax.legend(loc='upper right', frameon=False)


def _draw_dn_bar(ax, dn_df: pd.DataFrame | None, p: str):
    if dn_df is None or dn_df.empty:
        ax.text(0.5, 0.5, f'No DN segments for MMS{p}', ha='center', va='center')
        ax.set_axis_off()
        return
    ax.bar(dn_df['segment'], dn_df['DN_km'], color='#54a24b')
    ax.set_xlabel('Segment #')
    ax.set_ylabel('DN (km)')
    ax.grid(True, alpha=0.2)


def make_storyboards():
    for p in ['1','2','3','4']:
        df = _load_csv(p)
        dn_df = _load_dn_csv(p)
        dis_img, des_img = _load_spectrogram_images(p)

        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1.0, 0.6])

        # Row 1: DIS spectrogram image (or DES if desired side-by-side)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_axis_off()
        if dis_img is not None:
            ax1.imshow(dis_img)
            ax1.set_title(f'MMS{p} DIS spectrogram (omni)')
        else:
            ax1.text(0.5, 0.5, f'MMS{p} DIS spectrogram not available', ha='center', va='center')

        # Row 2: ViN overlay
        ax2 = fig.add_subplot(gs[1, 0])
        if df is not None and not df.empty:
            _draw_vin_overlay(ax2, df)
            ax2.set_title(f'MMS{p} ViN overlay')
        else:
            ax2.text(0.5, 0.5, f'MMS{p} ViN not available', ha='center', va='center')
            ax2.set_axis_off()

        # Row 3: DN bar chart
        ax3 = fig.add_subplot(gs[2, 0])
        _draw_dn_bar(ax3, dn_df, p)

        fig.tight_layout()
        out = EVENT_DIR / f'storyboard_mms{p}.png'
        fig.savefig(out, dpi=220)
        plt.close(fig)
        print(f'Wrote {out}')


if __name__ == '__main__':
    make_storyboards()

