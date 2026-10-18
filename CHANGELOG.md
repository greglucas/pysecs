# Changelog

## v0.5.0 (unreleased)

### New features

- Current fitting: `fit_J()` fits the SECS to observed horizontal sheet
  currents, decomposing them into divergence-free and curl-free parts and
  their field-aligned currents. `fit()` can also fit magnetic field and
  current observations jointly with the new keyword-only `J_loc`, `obs_J` and
  `J_std` arguments.
- FAC densities: `fac_density()` on `SECS` and `KalmanSECS` computes the
  field-aligned current density (and its variance) at each curl-free SEC
  from the fitted amplitudes and grid cell areas.

### Bug fixes

- The cached transfer matrices of `fit()` and `predict()` are recomputed when
  the location array passed in is modified in place between calls.

### Documentation

- New gallery example for current and joint magnetic field + current fits,
  and a theory section on fitting currents.

## v0.4.0 (2026-07-09)

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
