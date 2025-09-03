# Formation Metrics and Method Selection

This document summarizes the formation metrics used in `mms_mp.formation_detection` and how we recommend a timing method.

## Key Metrics

- Principal components (eigenvalues λ1 ≥ λ2 ≥ λ3 of covariance of centered positions)
  - Linearity = λ1 / (λ1 + λ2 + λ3)
  - Planarity = (λ1 + λ2) / (λ1 + λ2 + λ3)
  - Sphericity = λ3 / (λ1 + λ2 + λ3)
- Separations
  - Mean separation and standard deviation across MMS pairs
  - Separation uniformity = 1 − std/mean (higher is more uniform)
- Tetrahedron quality proxies
  - Approximate volume, edge statistics; additional proxy stored as `tetrahedron_quality_factor`

## Formation Types

- String-of-pearls
  - High linearity, very low sphericity
  - Optional axis-angle reinforcement: average angle from positions to the principal axis ≤ threshold
  - Quality metric: `string_quality = separation_uniformity`
- Linear
  - High linearity, modest constraints on sphericity
- Planar
  - High planarity, low sphericity
- Tetrahedral
  - Non-trivial volume and balanced eigenvalues
- Irregular/Collapsed
  - Fallback categories when thresholds are not met or separations are very small

## Thresholds (defaults)

- String-of-pearls
  - linearity > 0.85, sphericity < 0.10
  - axis-angle mean ≤ 12° (reinforcement; used to modulate confidence)
- Linear (fallback)
  - linearity > 0.75, sphericity < 0.20

These are module constants that can be tuned for specific campaigns.

## Recommended Timing Methods

- String-of-pearls → `string_of_pearls_timing`
- Linear → `linear_timing`
- Planar → `planar_timing`
- Tetrahedral → `tetrahedral_timing`
- Otherwise → `general_timing`

## Notes

- Confidence values are in [0,1]; stronger linearity and lower sphericity increase confidence for strings/linear.
- For strings, confidence is modulated by axis alignment to the principal axis when available.
- Always use MEC ephemeris as the authoritative source for determining positions and velocities.

