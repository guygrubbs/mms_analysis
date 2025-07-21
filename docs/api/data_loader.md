# data_loader Module

Data loading and CDF handling for MMS instruments.

## Functions

### `load_event(trange, probes=None, include_edp=False, include_hpca=True, cadence='fast')`

Load MMS data for a time range across multiple spacecraft.

**Parameters:**
- `trange` (list): `['YYYY-MM-DDTHH:MM', 'YYYY-MM-DDTHH:MM']`
- `probes` (list, optional): Spacecraft numbers `['1', '2', '3', '4']`. Default: all
- `include_edp` (bool): Include electric field data. Default: False
- `include_hpca` (bool): Include HPCA ion composition. Default: True
- `cadence` (str): `'fast'`, `'brst'`, or `'srvy'`. Default: 'fast'

**Returns:**
- `dict`: `{probe: {variable: (time, data), ...}, ...}`

**Example:**
```python
data = load_event(['2019-11-12T04:00', '2019-11-12T05:00'], 
                  probes=['1', '2'], include_edp=True)
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
| `N_he` | He⁺ density | cm⁻³ | HPCA |
| `V_i_gse` | Ion bulk velocity | km/s | FPI-DIS |
| `V_e_gse` | Electron bulk velocity | km/s | FPI-DES |
| `V_he_gsm` | He⁺ bulk velocity | km/s | HPCA |
| `E_gse` | Electric field | mV/m | EDP |
| `POS_gsm` | Spacecraft position | km | Ephemeris |

## Error Handling

- Missing instruments fall back to survey mode
- Unavailable variables filled with NaN
- Quality flags automatically applied