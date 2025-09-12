"""
Build a multi-page PDF figure pack for the 2019-01-27 event from generated PNGs.
Input folder: results/events/2019-01-27_1215-1255
Output: results/events/2019-01-27_1215-1255/figure_pack.pdf
"""
from __future__ import annotations
import pathlib
from PIL import Image

EVENT_DIR = pathlib.Path('results/events_pub/2019-01-27_1215-1255')

# Ordered pages (add if they exist)
PAGE_FILES = [
    'vn_overlay_stack.png',
    'vn_overlay_mms1.png', 'vn_diff_hist_mms1.png', 'vn_offset_hist_mms1.png',
    'vn_overlay_mms2.png', 'vn_diff_hist_mms2.png', 'vn_offset_hist_mms2.png',
    'vn_overlay_mms3.png', 'vn_diff_hist_mms3.png', 'vn_offset_hist_mms3.png',
    'vn_overlay_mms4.png', 'vn_diff_hist_mms4.png', 'vn_offset_hist_mms4.png',
    'mms1_DIS_omni.png', 'mms1_DES_omni.png',
    'mms2_DIS_omni.png', 'mms2_DES_omni.png',
    'mms3_DIS_omni.png', 'mms3_DES_omni.png',
    'mms4_DIS_omni.png', 'mms4_DES_omni.png',
    'mms1_DN_segments.png', 'mms2_DN_segments.png', 'mms3_DN_segments.png', 'mms4_DN_segments.png',
]

imgs = []
for name in PAGE_FILES:
    f = EVENT_DIR / name
    if f.exists():
        img = Image.open(f).convert('RGB')
        imgs.append(img)

if not imgs:
    raise SystemExit('No images found to build figure pack; run make_event_figures_20190127.py first.')

pdf_path = EVENT_DIR / 'figure_pack.pdf'
first, rest = imgs[0], imgs[1:]
first.save(pdf_path, save_all=True, append_images=rest)
print(f'Wrote {pdf_path.resolve()} with {1+len(rest)} pages')

