"""Temporal state-space estimation of SECS amplitudes.

Snapshot SECS fits treat every time step independently, so the solution
can jump between steps whenever the data change slightly (station
dropouts, noise, localized disturbances). This module couples the time
steps with a linear-Gaussian state-space model on the SEC amplitudes and
estimates them with a Kalman filter and Rauch-Tung-Striebel smoother:

- state evolution: a first-order autoregressive (Ornstein-Uhlenbeck)
  model ``m_t = phi_t m_{t-1} + w_t`` with ``phi_t = exp(-dt/tau)``,
  which relaxes toward zero (quiet conditions) with correlation time
  ``tau`` and stationary variance ``prior_std**2``
- observations: the SECS magnetic field transfer matrix with
  independent Gaussian errors given by ``obs_std``

The stationary prior plays the role that the truncated-SVD ``epsilon``
plays in the snapshot fit, but it is applied consistently at every time
step, which removes the step-to-step flicker caused by the retained
singular subspace changing between snapshots. Innovation gating
optionally downweights observations that are inconsistent with the
recent past (e.g. a spike at a single station) before they can steer
the entire solution.

This follows the temporal-regularization strategies reviewed for
ionospheric data assimilation by Laundal et al. (2025),
doi:10.1007/s10712-025-09918-3, and used in the Lompe technique
(Laundal et al., 2022, doi:10.1029/2022JA030356).
"""

import numpy as np

from pysecs.secs import SECS


__all__ = ["KalmanSECS"]


class KalmanSECS:
    """Kalman filter / smoother for time series of SECS amplitudes.

    Parameters
    ----------
    secs : SECS
        The elementary current system holding the SEC locations. Its
        transfer matrices are reused for the state-space model and the
        fitted amplitudes are stored back into it for predictions.

    tau : float
        Temporal correlation time of the amplitudes, in the same units
        as the spacing of the ``times`` passed to :meth:`fit` (seconds
        if ``times`` is a datetime64 array). Amplitudes decorrelate as
        ``exp(-dt / tau)``; larger values produce smoother time series.

    prior_std : float, optional
        Stationary (a priori) standard deviation of each SEC amplitude
        in Amperes. This regularizes the spatial inversion the same way
        ``epsilon`` regularizes the snapshot fit: amplitudes the data do
        not constrain relax to zero with this uncertainty. When None, it
        is estimated from the RMS amplitude of an ordinary snapshot fit
        to the same observations.
        Default: None

    gate_sigma : float, optional
        Innovation gating threshold in standard deviations. At each time
        step, observations whose innovations (measurement minus model
        forecast) exceed ``gate_sigma`` standardized deviations have
        their errors inflated so they cannot dominate the update. This
        rejects impulsive single-station disturbances while leaving
        consistent large-scale changes alone. None disables gating.
        Default: None

    smoother : bool
        Whether to run the backward Rauch-Tung-Striebel smoother after
        the forward filter pass. The smoother uses future as well as
        past data at each time step (non-causal) and is recommended for
        post-processing; disable for a purely causal (real-time) filter.
        Default: True

    epsilon, mode :
        Passed to the snapshot ``SECS.fit()`` used only to estimate
        ``prior_std`` when it is not given.

    Notes
    -----
    The full amplitude covariance is stored for every time step in
    ``sec_amps_cov`` with shape (ntimes, nsec, nsec), which requires
    ``8 * ntimes * nsec**2`` bytes of memory (about 0.5 GB for 2000 time
    steps of 180 SECs). Keep the SEC grid and analysis windows sized
    accordingly.

    References
    ----------
    .. [1] Kalman, R. E. "A New Approach to Linear Filtering and
        Prediction Problems." J. Basic Eng. 82.1 (1960): 35-45.
    .. [2] Rauch, H. E., F. Tung, and C. T. Striebel. "Maximum likelihood
        estimates of linear dynamic systems." AIAA Journal 3.8 (1965).
    .. [3] Laundal, K. M., et al. "Next-Generation Data Assimilation
        Methods for Polar Ionospheric Electrodynamics." Surveys in
        Geophysics (2025). doi:10.1007/s10712-025-09918-3
    """

    def __init__(
        self,
        secs: SECS,
        tau: float,
        prior_std: float | None = None,
        gate_sigma: float | None = None,
        smoother: bool = True,
        epsilon: float = 0.05,
        mode: str = "relative",
    ) -> None:
        if tau <= 0:
            raise ValueError("tau must be positive")
        if prior_std is not None and prior_std <= 0:
            raise ValueError("prior_std must be positive")
        if gate_sigma is not None and gate_sigma <= 0:
            raise ValueError("gate_sigma must be positive")

        self.secs = secs
        self.tau = tau
        self.prior_std = prior_std
        self.gate_sigma = gate_sigma
        self.smoother = smoother
        self.epsilon = epsilon
        self.mode = mode

        self.sec_amps = np.empty((0, secs.nsec))
        self.sec_amps_var = np.empty((0, secs.nsec))
        self.sec_amps_cov = np.empty((0, secs.nsec, secs.nsec))

    @staticmethod
    def _delta_seconds(times: np.ndarray) -> np.ndarray:
        """Return the time differences of ``times`` as floats."""
        times = np.asarray(times)
        if np.issubdtype(times.dtype, np.datetime64):
            dt = np.diff(times) / np.timedelta64(1, "s")
        else:
            dt = np.diff(times.astype(float))
        return dt

    def fit(
        self,
        obs_loc: np.ndarray,
        obs_B: np.ndarray,
        times: np.ndarray,
        obs_std: np.ndarray | None = None,
    ) -> "KalmanSECS":
        """Run the filter (and smoother) over a time series of observations.

        Parameters
        ----------
        obs_loc : ndarray (nobs, 3 [lat, lon, r])
            Observation locations, fixed over the time series.

        obs_B : ndarray (ntimes, nobs, 3 [Bx, By, Bz])
            Observed magnetic fields. Non-finite values are treated as
            missing data at that time step.

        times : ndarray (ntimes,)
            Strictly increasing observation times, either as floats (in
            the same units as ``tau``) or as np.datetime64 (``tau`` in
            seconds).

        obs_std : ndarray (ntimes, nobs, 3), optional
            1-sigma standard errors of the observations. Infinite values
            eliminate an observation at that time step, e.g. for station
            dropouts. Default: ones (equal weights).
        """
        if obs_B.ndim == 2:
            obs_B = obs_B[np.newaxis, ...]
        if obs_std is None:
            obs_std = np.ones_like(obs_B)

        ntimes = len(obs_B)
        times = np.asarray(times)
        if times.shape != (ntimes,):
            raise ValueError("times must be 1-D with one entry per time step")
        dt = self._delta_seconds(times)
        if np.any(dt <= 0):
            raise ValueError("times must be strictly increasing")

        obs_B_flat = obs_B.reshape(ntimes, -1)
        obs_std_flat = obs_std.reshape(ntimes, -1).astype(float)

        # Observation operator from the SECS transfer functions
        H_full = self.secs._calc_T(obs_loc).reshape(-1, self.secs.nsec)

        prior_std = self.prior_std
        if prior_std is None:
            # Estimate the amplitude scale from an ordinary snapshot fit
            self.secs.fit(
                obs_loc, obs_B, obs_std=obs_std, epsilon=self.epsilon, mode=self.mode
            )
            rms = np.sqrt(np.mean(self.secs.sec_amps**2))
            prior_std = rms if rms > 0 else 1.0

        nsec = self.secs.nsec
        prior_var = prior_std**2
        phi = np.exp(-dt / self.tau)

        # Forward Kalman filter
        m = np.zeros(nsec)
        P = prior_var * np.eye(nsec)
        m_f = np.empty((ntimes, nsec))
        P_f = np.empty((ntimes, nsec, nsec))
        for t in range(ntimes):
            if t > 0:
                # Ornstein-Uhlenbeck (AR(1)) forecast: exact discretization
                # with process noise (1 - phi**2) * prior_var keeping the
                # stationary variance at prior_var for any time spacing
                m = phi[t - 1] * m
                P = phi[t - 1] ** 2 * P
                P[np.diag_indices_from(P)] += (1 - phi[t - 1] ** 2) * prior_var

            m, P = self._measurement_update(
                m, P, H_full, obs_B_flat[t], obs_std_flat[t]
            )
            m_f[t] = m
            P_f[t] = P

        if self.smoother and ntimes > 1:
            m_f, P_f = self._rts_smoother(m_f, P_f, phi, prior_var)

        self.sec_amps = m_f
        self.sec_amps_cov = P_f
        self.sec_amps_var = np.einsum("tii->ti", P_f).copy()

        # Store the amplitudes in the SECS object so its (cached)
        # prediction machinery can be reused
        self.secs.sec_amps = self.sec_amps
        self.secs.sec_amps_var = self.sec_amps_var
        # The snapshot inversion operators do not describe these
        # amplitudes; prediction variances go through this class instead
        self.secs._VWU_patterns = None
        self.secs._pattern_index = None
        return self

    @staticmethod
    def _rts_smoother(
        m_f: np.ndarray, P_f: np.ndarray, phi: np.ndarray, prior_var: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the backward Rauch-Tung-Striebel smoothing pass."""
        m_s = np.empty_like(m_f)
        P_s = np.empty_like(P_f)
        m_s[-1] = m_f[-1]
        P_s[-1] = P_f[-1]
        for t in range(len(m_f) - 2, -1, -1):
            # One-step forecast from t to t+1
            P_pred = phi[t] ** 2 * P_f[t]
            P_pred[np.diag_indices_from(P_pred)] += (1 - phi[t] ** 2) * prior_var
            # Smoother gain G = phi * P_f P_pred^-1 (P_pred is SPD)
            G = phi[t] * np.linalg.solve(P_pred, P_f[t]).T
            m_s[t] = m_f[t] + G @ (m_s[t + 1] - phi[t] * m_f[t])
            P_s[t] = P_f[t] + G @ (P_s[t + 1] - P_pred) @ G.T
        return m_s, P_s

    def _measurement_update(
        self,
        m: np.ndarray,
        P: np.ndarray,
        H_full: np.ndarray,
        z_row: np.ndarray,
        std_row: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Kalman measurement update with optional innovation gating."""
        valid = np.isfinite(std_row) & np.isfinite(z_row)
        if not np.any(valid):
            # Nothing observed at this time: pure forecast
            return m, P

        H = H_full[valid]
        z = z_row[valid]
        rstd = std_row[valid].copy()

        while True:
            PHt = P @ H.T
            S = H @ PHt
            S[np.diag_indices_from(S)] += rstd**2
            nu = z - H @ m

            if self.gate_sigma is not None:
                # Standardized innovations flag observations that are
                # inconsistent with the forecast plus its uncertainty
                u = np.abs(nu) / np.sqrt(np.diag(S))
                flagged = u > self.gate_sigma
                if np.any(flagged):
                    # Inflate the errors just enough to bring the flagged
                    # innovations back to the gate threshold, then redo
                    # the update with the new error covariance
                    rstd[flagged] *= u[flagged] / self.gate_sigma
                    continue
            break

        K = np.linalg.solve(S, PHt.T).T
        m = m + K @ nu
        # Joseph form for numerical stability (keeps P symmetric PSD)
        ImKH = np.eye(len(m)) - K @ H
        P = ImKH @ P @ ImKH.T + (K * rstd**2) @ K.T
        P = 0.5 * (P + P.T)
        return m, P

    def predict(
        self, pred_loc: np.ndarray, J: bool = False, return_var: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict magnetic fields or currents from the estimated amplitudes.

        Parameters
        ----------
        pred_loc : ndarray (npred, 3 [lat, lon, r])
            Locations where the predictions are desired.

        J : bool
            Whether to predict currents (True) or magnetic fields (False).
            Default: False

        return_var : bool
            Whether to also return prediction variances propagated from
            the full amplitude covariance of the filter/smoother.
            Default: False

        Returns
        -------
        ndarray (ntimes, npred, 3)
            Predicted values; a (predictions, variances) tuple if
            ``return_var`` is True.
        """
        if return_var and len(self.sec_amps_cov) == 0:
            raise ValueError("Call fit() before predicting with return_var=True.")

        pred = self.secs.predict(pred_loc, J=J)
        if not return_var:
            return pred

        T_pred = self.secs._T_pred_J if J else self.secs._T_pred_B
        T_flat = T_pred.reshape(-1, self.secs.nsec)
        ntimes = len(self.sec_amps)
        npred = len(pred_loc)
        var = np.empty((ntimes, npred, 3))
        for t in range(ntimes):
            # diag(T P T^T) without forming the full product
            var[t] = np.sum((T_flat @ self.sec_amps_cov[t]) * T_flat, axis=1).reshape(
                npred, 3
            )
        return pred, np.squeeze(var)

    def predict_B(
        self, pred_loc: np.ndarray, return_var: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict magnetic fields; see :meth:`predict`."""
        return self.predict(pred_loc, return_var=return_var)

    def predict_J(
        self, pred_loc: np.ndarray, return_var: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict currents; see :meth:`predict`."""
        return self.predict(pred_loc, J=True, return_var=return_var)
