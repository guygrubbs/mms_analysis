# MMS Magnetopause Analysis Toolkit ( `mms_mp` )

A **Python-only**, research-grade toolkit that reproduces—and extends—the IDL workflow used to study
magnetopause reconnection events with **NASA’s Magnetospheric Multiscale (MMS)** mission data.  
It loads Level-2 CDFs directly from CDAWeb, detects boundary crossings with
multi-parameter logic, reconstructs boundary motion, performs
multi-spacecraft timing, and produces publication-quality figures.

| Feature | Status |
|---------|:------:|
| Automatic CDF download (FGM / FPI / HPCA / EDP / ephemeris) | ✅ |
| Hybrid LMN (MVA + Shue model fallback) | ✅ |
| Cold-ion + \(B_N\) boundary detector (hysteresis) | ✅ |
| E×B drift + He⁺ bulk blending | ✅ |
| Displacement integration ±σ | ✅ |
| 2-to-4 SC timing → \( \hat{n},\;V_{\text{phase}} \) ±σ | ✅ |
| One-command CLI + CSV / JSON / PNG output | ✅ |
| Notebook-friendly `main_analysis.py` demo | ✅ |

---

## 1  Quick Start

```bash
# ❶ Create a clean environment
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install --upgrade pip

# ❷ Install dependencies
pip install -r requirements.txt

# ❸ Run the high-level demo (1-hour event on 2019-11-12)
python main_analysis.py
#   – interactive plots pop up
#   – results printed to console

# ❹ Full CLI (results into ./results/)
python -m mms_mp.cli \
    --start 2019-11-12T04:00 --end 2019-11-12T05:00 \
    --probes 1 2 3 4 --plot
````

> **Tip:** First run might take a few minutes because CDFs are downloaded
> from CDAWeb to `~/.pyspedas/`.

---

## 2  Repository Layout

```
mms_mp/                 Core package
│
├── __init__.py         • Package initialization and imports
├── data_loader.py      • CDF download + variable extraction
├── coords.py           • LMN transforms (MVA / model / hybrid)
├── resample.py         • Multi-var resampling/merging helpers
├── electric.py         • E×B drift, v_N selection
├── quality.py          • Instrument quality-flag masks
├── boundary.py         • Multi-parameter boundary detector
├── motion.py           • Integrate v_N → displacement  ±σ
├── multispacecraft.py  • Timing method (n̂, V_ph) + alignment
├── visualize.py        • Publication-ready plotting helpers
├── spectra.py          • FPI ion/electron spectrograms
├── thickness.py        • Layer thickness calculation utilities
└── cli.py              • Command-line pipeline
main_analysis.py        Notebook/demo script
requirements.txt        Core dependencies
pyproject.toml          Package configuration
README.md
LICENSE
```

---

## 3  Module Overview

| Module            | Highlight                                                                                  |
| ----------------- | ------------------------------------------------------------------------------------------ |
| `data_loader`     | 1-call loader for FGM, FPI, HPCA, EDP, ephemeris + pandas/NumPy friendly output            |
| `coords`          | `hybrid_lmn()` → returns MVA result if eigen-ratio ≥ 5, else Shue (1997) model normal      |
| `boundary`        | State-machine uses He⁺ density **and** $B_N$ rotation; hysteresis thresholds avoid noise   |
| `electric`        | `exb_velocity_sync()` merges E & B to compute drift; `normal_velocity()` blends ExB + bulk |
| `motion`          | `integrate_disp()` supports trapezoid / Simpson; propagates 1-σ if velocity error supplied |
| `multispacecraft` | SVD timing solver (2–4 SC) returns n̂, Vₚₕ ±σ; `stack_aligned()` builds overlay arrays     |
| `visualize`       | `summary_single()` (4-panel quick-look), `overlay_multi()`, `plot_displacement()`          |
| `thickness`       | `layer_thicknesses()` calculates magnetopause layer thickness from displacement data       |
| `cli`             | Downloads, analyses, saves JSON + CSV + figures; ideal for batch runs / cron jobs          |

---

## 4  Typical Workflow (Concept Map)

```
              ┌──────────────┐   LMN  ┌───────────────┐
raw CDFs ───▶ │ data_loader  │ ─────▶ │  coords       │
(FGM,FPI,HPCA)└──────────────┘        └───────────────┘
     │                               ↙
     │ E,B,V,N → merge_vars()        │ R (3×3)
     ▼                               │
┌──────────────┐ He⁺, B_N  ┌─────────┴─────────┐
│ resample     │──────────▶│ boundary detector │─┐
└──────────────┘           └─────────┬─────────┘ │ layers
                                      │           │
        E×B+bulk                      ▼           │
     ┌──────────────┐  v_N  ┌────────────────────┘
     │ electric     │──────▶│ motion.integrate_disp ──►  Δs(t), σ
     └──────────────┘       └────────────────────┐
                                                  │ layer thickness
       positions + t_cross        ┌───────────────┘
       ───────────────────────────▶│ multispacecraft
                                   └───────────────┘  (n̂, V_ph, σ)
```

---

## 5  Outputs

| CLI flag   | Result                                                                      |
| ---------- | --------------------------------------------------------------------------- |
| `--plot`   | PNG quick-look figures in `results/figures/`                                |
| *(always)* | `results/mp_analysis.json` — crossings, positions, n̂, Vₚₕ, layer thickness |
| *(always)* | `results/layer_thickness.csv`                                               |

---

## 6  Dependencies

| Library              | Purpose                                   |
| -------------------- | ----------------------------------------- |
| **pySPEDAS ≥ 1.7.20** | Download & read MMS CDFs                  |
| NumPy, SciPy, pandas | Numerics, resampling, Simpson integration | (optional) |
| Matplotlib           | Plots + spectrograms                      | (optional) |
| (Optional) tqdm      | Progress bars (not required)              | (optional) |

Install everything via:

```bash
pip install -r requirements.txt
```

---

## 7  Citing / References

If you use this toolkit in a publication, please cite the underlying data and methods:

* **MMS Mission:** Burch et al., *Space Sci. Rev.* (2016)
* **Hybrid LMN idea:** Denton et al., *JGR Space Physics* (2018)
* **Cold He⁺ boundary tracking:** Llera et al., *JGR Space Physics* (2023)

A BibTeX snippet is provided in `docs/citation.bib`.

---

## 8  Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-improvement`)
3. Add / edit code + **unit tests** in `tests/`
4. Submit a pull request

We follow **PEP-8** with `black` auto-formatter (line length = 88).

---

## 9  License

This project is released under the **MIT License**.
See [LICENSE](LICENSE) for details.

Enjoy exploring the magnetopause!
*— mms\_mp maintainers*
