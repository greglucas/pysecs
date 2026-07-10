# Changelog

## v0.4.0 (unreleased)

### New features

- Curl-free SECS support: implemented the curl-free magnetic field transfer
  function `T_cf` and corrected the field-aligned current directions in `J_cf`.
- Robust fitting: added iteratively reweighted least squares (IRLS) to
  down-weight outlying observations during `fit()`.
- `KalmanSECS`: a temporal state-space filter and smoother for time-evolving
  SECS solutions.
- Automatic grid generation: `make_grid()` builds a SECS pole grid directly
  from an observation network.

### Bug fixes

- Corrected `sec_amps_var` scaling and added prediction variances.
- Replaced the arccos angular-distance formula with an atan2/hypot form to
  avoid numerical issues near coincident points.

### Performance

- Grouped repeated uncertainty patterns in `fit()` and added a benchmark
  suite.

### Documentation

- New theory page, gallery examples for grid generation, robust vs. Kalman
  fitting comparisons, and expanded API documentation.

### Maintenance

- Added support for Python 3.14.
