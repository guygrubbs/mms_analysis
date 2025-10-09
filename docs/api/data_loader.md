# data_loader Module

Data loading and CDF handling for MMS instruments.

## Functions

### `load_event(trange, probes=None, data_rate_fgm='fast', data_rate_fpi='fast', data_rate_hpca='fast', include_hpca=True, include_edp=False, include_ephem=True, include_brst=False, include_srvy=False, include_slow=False, download_only=False)`

Load MMS data for a time range across multiple spacecraft. The loader records
provenance for every harvested variable so publication workflows can trace which
CDAWeb products were used.

**Parameters:**
- `trange` (list[str]): `['YYYY-MM-DDTHH:MM[:SS]', 'YYYY-MM-DDTHH:MM[:SS]']`
- `probes` (list[str], optional): Spacecraft numbers `['1', '2', '3', '4']`. Default: all.
- `data_rate_fgm` (str): Preferred FGM cadence (`'fast'`, `'brst'`, `'srvy'`).
- `data_rate_fpi` (str): Preferred FPI cadence for both DIS and DES moments.
- `data_rate_hpca` (str): Preferred HPCA cadence when `include_hpca=True`.
- `include_hpca` (bool): Load HPCA moments. When `True`, the loader will fall back to
  QL products and perform cross-spacecraft reconstruction so that He⁺ densities and
  velocities remain physically meaningful even when MMS4 lacks burst coverage.
- `include_edp` (bool): Load EDP electric field and spacecraft potential data.
- `include_ephem` (bool): Load MEC (preferred) or definitive ephemeris.
- `include_brst`, `include_srvy`, `include_slow` (bool): Permit additional cadences
to be attempted after the preferred rate when files are missing.
- `download_only` (bool): Download CDFs without harvesting variables.

**Returns:**
- `dict`: `{probe: {variable: (time, data), ...}, '__meta__': provenance}`

`__meta__` keys:

| Field | Description |
|-------|-------------|
| `requested_trange` | ISO-formatted start/end supplied to the loader |
| `probes` | Spacecraft identifiers included in the request |
| `cadence_preferences` | Ordered cadences attempted per instrument |
| `download_summary` | Retry history and selected cadence per instrument |
| `ephemeris_sources` | `'mec'`, `'definitive'`, `'unavailable'`, or `'skipped'` for each probe |
| `sources` | Mapping from logical variable keys to CDAWeb variable names (or `'skipped'`/`None`) |
| `time_coverage` | Observed start/end per variable in UTC |
| `warnings` | Notes about coverage deviations or reconstructed quantities |

Quality flags (`mms{n}_dis_quality_flag`, `mms{n}_des_quality_flag`, and
`mms{n}_hpca_status_flag` when HPCA is enabled) are harvested automatically when
present so that downstream quality-mask helpers do not need to query CDAWeb again.
When burst-level coverage is missing (e.g., MMS4 electrons after 2018), the loader
automatically retries with QL moments and records the successful fallback in the
`download_summary` map.

**Example:**
```python
evt = load_event(
    ['2019-11-12T04:00', '2019-11-12T05:00'],
    probes=['1', '2'],
    data_rate_fpi='fast',
    include_hpca=True,
    include_edp=True,
)

meta = evt['__meta__']
print(meta['download_summary']['fgm'])
```

### `load_ephemeris(trange, probes=None)`

Load spacecraft position and attitude data.

**Parameters:**
- `trange` (list): Time range
- `probes` (list, optional): Spacecraft numbers

**Returns:**
- `dict`: Position and velocity data in GSM coordinates

### Variable Keys

| Key | Description | Units | Source |
|-----|-------------|-------|--------|
| `B_gsm` | Magnetic field | nT | FGM |
| `N_e` | Electron density | cm⁻³ | FPI-DES |
| `N_tot` | Total ion density | cm⁻³ | FPI-DIS |
| `N_he` | He⁺ density | cm⁻³ | HPCA (QL fallback + reconstruction as needed) |
| `V_i_gse` | Ion bulk velocity | km/s | FPI-DIS |
| `V_e_gse` | Electron bulk velocity | km/s | FPI-DES |
| `V_he_gsm` | He⁺ bulk velocity | km/s | HPCA (QL fallback + reconstruction as needed) |
| `E_gse` | Electric field | mV/m | EDP |
| `POS_gsm` | Spacecraft position | km | MEC/definitive ephemeris |
| `VEL_gsm` | Spacecraft velocity | km/s | MEC/definitive ephemeris |
| `mms{n}_dis_quality_flag` | Ion quality flag | dimensionless | FPI-DIS |
| `mms{n}_des_quality_flag` | Electron quality flag | dimensionless | FPI-DES |
| `mms{n}_hpca_status_flag` | HPCA status flag | bit mask | HPCA |

## Error Handling

- Each requested cadence is retried up to three times with light exponential back-off.
- When burst-level products are unavailable, the loader automatically retries with QL
  equivalents and records the attempt in `download_summary` (e.g., `fpi_dis_ql`).
- If a spacecraft is still missing a quantity after QL fallback, the loader reconstructs
  it from the remaining probes and documents the provenance in `__meta__['sources']`.
  Ion and electron densities fall back on quasi-neutrality (N$_e$ ≈ N$_{tot}$) before
  an error is raised.
- Quality/status flags are harvested when available to aid downstream QC.
