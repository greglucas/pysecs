"""Spherical Elementary Current System (SECS) module.

Calculate magnetic field transfer functions and fit a system (SECS) to observations.
"""

import warnings

import numpy as np


__all__ = [
    "SECS",
    "J_cf",
    "J_df",
    "T_cf",
    "T_df",
    "calc_angular_distance",
    "calc_bearing",
]


class SECS:
    """Spherical Elementary Current System (SECS).

    The algorithm is implemented directly in spherical coordinates
    from the equations of the 1999 Amm & Viljanen paper [1]_. The
    magnetic field of the curl-free system is implemented from
    Equation 2.15 of the Vanhamäki & Juusola review chapter [2]_.

    Parameters
    ----------
    sec_df_loc : ndarray (nsec, 3 [lat, lon, r])
        The latitude, longiutde, and radius of the divergence free (df) SEC locations.

    sec_cf_loc : ndarray (nsec, 3 [lat, lon, r])
        The latitude, longiutde, and radius of the curl free (cf) SEC locations.

    References
    ----------
    .. [1] Amm, O., and A. Viljanen. "Ionospheric disturbance magnetic field
        continuation from the ground to the ionosphere using spherical elementary
        current systems." Earth, Planets and Space 51.6 (1999): 431-440.
        doi:10.1186/BF03352247
    .. [2] Vanhamäki, H., and L. Juusola. "Introduction to Spherical Elementary
        Current Systems." Ionospheric Multi-Spacecraft Analysis Tools,
        ISSI Scientific Report Series 17 (2020): 5-33.
        doi:10.1007/978-3-030-26732-2_2
    """

    def __init__(
        self, sec_df_loc: np.ndarray | None = None, sec_cf_loc: np.ndarray | None = None
    ) -> None:
        if sec_df_loc is None and sec_cf_loc is None:
            raise ValueError("Must specify at least one of sec_df_loc of sec_cf_loc")

        if sec_df_loc is None:
            sec_df_loc = np.empty((0, 3))
        # Convert to array if not already and
        # add an empty dimension if only one SEC location is passed in
        self.sec_df_loc = np.atleast_2d(sec_df_loc)
        if self.sec_df_loc.shape[-1] != 3:
            raise ValueError("SEC DF locations must have 3 columns (lat, lon, r)")

        if sec_cf_loc is None:
            sec_cf_loc = np.empty((0, 3))
        # Convert to array if not already and
        # add an empty dimension if only one SEC location is passed in
        self.sec_cf_loc = np.atleast_2d(sec_cf_loc)
        if self.sec_cf_loc.shape[-1] != 3:
            raise ValueError("SEC CF locations must have 3 columns (lat, lon, r)")

        # Storage of the scaling factors
        self.sec_amps = np.empty((0, self.nsec))
        self.sec_amps_var = np.empty((0, self.nsec))

        # Keep some values around for cache lookups. Empty arrays are
        # the "not computed yet" sentinels: they never compare equal to
        # real location arrays, so the first use fills the cache.
        self._obs_loc = np.empty((0, 3))
        self._T_obs_flat = np.empty((0, self.nsec))
        self._pred_loc_B = np.empty((0, 3))
        self._T_pred_B = np.empty((0, 3, self.nsec))
        self._pred_loc_J = np.empty((0, 3))
        self._T_pred_J = np.empty((0, 3, self.nsec))

        # Inversion operators from the most recent fit(), used to
        # propagate amplitude covariances to prediction variances
        self._VWU_patterns: list[np.ndarray] | None = None
        self._pattern_index: np.ndarray | None = None

    @classmethod
    def from_observations(
        cls,
        obs_loc: np.ndarray,
        r_shell: float,
        spacing: float | tuple[float, float] | None = None,
        padding: float | None = None,
        min_distance: float | None = None,
        df: bool = True,
        cf: bool = False,
    ) -> "SECS":
        """Build a SECS with an automatically generated grid.

        Lays out a regular divergence-free and/or curl-free grid
        covering the given observation locations with
        :func:`pysecs.make_grid` and constructs a SECS on it. This is a
        convenience constructor for the common case of a single grid at
        a single shell radius; call :func:`pysecs.make_grid` directly
        for more control (e.g. independent df/cf grids, or an
        additional image shell via :func:`pysecs.make_image_grid`).

        Parameters
        ----------
        obs_loc : ndarray (nobs, 3 [lat, lon, r])
            The observation locations the grid should cover.

        r_shell : float
            The radius of the SEC shell.

        spacing, padding, min_distance :
            Passed to :func:`pysecs.make_grid`.

        df : bool
            Whether to place divergence-free SECs on the grid.
            Default: True

        cf : bool
            Whether to place curl-free SECs on the grid.
            Default: False

        Returns
        -------
        SECS
            A SECS constructed on the generated grid.
        """
        if not df and not cf:
            raise ValueError("Must request at least one of df or cf")

        # Deferred to avoid a circular import: pysecs.grids imports
        # calc_angular_distance from this module at load time.
        from pysecs.grids import make_grid  # noqa: PLC0415

        grid = make_grid(
            obs_loc,
            r_shell,
            spacing=spacing,
            padding=padding,
            min_distance=min_distance,
        )
        return cls(
            sec_df_loc=grid if df else None, sec_cf_loc=grid if cf else None
        )

    @property
    def has_df(self) -> bool:
        """Whether this system has any divergence free currents."""
        return len(self.sec_df_loc) > 0

    @property
    def has_cf(self) -> bool:
        """Whether this system has any curl free currents."""
        return len(self.sec_cf_loc) > 0

    @property
    def nsec(self) -> int:
        """The number of elementary currents in this system."""
        return len(self.sec_df_loc) + len(self.sec_cf_loc)

    @staticmethod
    def _compute_VWU(
        T_obs_flat: np.ndarray, std_flat: np.ndarray, epsilon: float, mode: str
    ) -> np.ndarray:
        """Compute the VWU matrix from the SVD of the transfer function.

        This function computes the VWU matrix from the SVD of the transfer function
        and filters the singular values based on the specified mode. It is broken out
        to allow for easier branching logic in the fit() function.

        Parameters
        ----------
        T_obs_flat : ndarray
            The flattened transfer function matrix.
        std_flat : ndarray
            The flattened standard deviation matrix.
        epsilon : float
            The threshold for filtering singular values.
        mode : str
            The mode for filtering singular values.
            Options are 'relative' or 'variance'.

        Returns
        -------
        ndarray
            The VWU matrix.
        """
        # Weight the design matrix
        weighted_T = T_obs_flat / std_flat[:, np.newaxis]

        # SVD
        U, S, Vh = np.linalg.svd(weighted_T, full_matrices=False)

        # Filter components
        if mode == "relative":
            valid = S >= epsilon * S.max()
        elif mode == "variance":
            energy = np.cumsum(S**2)
            total = energy[-1]
            threshold = np.searchsorted(energy / total, 1 - epsilon) + 1
            valid = np.arange(len(S)) < threshold
        else:
            raise ValueError(f"Unknown SVD filtering mode: '{mode}'")

        # Never invert exactly-zero singular values. These arise when the
        # transfer matrix has all-zero columns, e.g. curl-free SECs with
        # observations only below the current shell where their magnetic
        # field vanishes identically.
        valid &= S > 0

        # Truncate and build VWU
        U = U[:, valid]
        S = S[valid]
        Vh = Vh[valid, :]
        W = 1.0 / S

        # Scale the rows of Vh by W via broadcasting rather than
        # materializing the dense diagonal matrix diag(W)
        return (Vh.T * W) @ U.T

    def fit(
        self,
        obs_loc: np.ndarray,
        obs_B: np.ndarray,
        obs_std: np.ndarray | None = None,
        epsilon: float = 0.05,
        mode: str = "relative",
        robust: str | None = None,
        robust_maxiter: int = 20,
        robust_tol: float = 1e-6,
    ) -> "SECS":
        """Fits the SECS to the given observations.

        Given a number of observation locations and measurements,
        this function fits the SEC system to them. It uses singular
        value decomposition (SVD) to fit the SEC amplitudes with the
        `epsilon` parameter used to regularize the solution.

        Parameters
        ----------
        obs_locs : ndarray (nobs, 3 [lat, lon, r])
            Contains latitude, longitude, and radius of the observation locations
            (place where the measurements are made)

        obs_B: ndarray (ntimes, nobs, 3 [Bx, By, Bz])
            An array containing the measured/observed B-fields.

        obs_std : ndarray (ntimes, nobs, 3 [stdX, stdY, stdZ]), optional
            Standard error (1-sigma) of vector components at each observation
            location. This can be used to weight different observations
            more/less heavily. An infinite value eliminates the observation
            from the fit.
            Default: ones(nobs, 3) equal weights

        epsilon : float
            Value used to regularize/smooth the SECS amplitudes. Epsilon  has
            different meanings depending on the mode used, described in that
            parameter section. Must be between 0 and 1. A higher number
            produces a more regularized (smoother) solution.
            Default: 0.05

        mode : str
            The mode used to filter the singular values. Options are:

            - 'relative': filter singular values that are less than epsilon times
              the largest singular value, keep [S >= epsilon * S.max()].
            - 'variance': filter singular values that contribute to 1-epsilon of
              the total energy of the system (% total variance as a ratio).

            Default: 'relative'

        robust : str, optional
            Robust reweighting of the observations. A localized disturbance
            at a single station (spikes, baseline jumps, instrument noise)
            is inconsistent with any smooth current system this basis can
            produce, so it shows up as a large standardized residual and is
            automatically downweighted by iteratively reweighted least
            squares (IRLS). Weights are applied per vector component and
            per time step. Options are:

            - None: ordinary weighted least squares (default).
            - 'huber': Huber weights, downweights outliers gradually.
            - 'bisquare': Tukey biweight, fully rejects gross outliers.

        robust_maxiter : int
            Maximum number of IRLS iterations. Each iteration refits every
            time step with its own weights, so robust fits of long time
            series cost roughly ``robust_maxiter`` times the ordinary fit.
            Default: 20

        robust_tol : float
            Relative amplitude change used to declare IRLS convergence.
            Default: 1e-6
        """
        if obs_loc.shape[-1] != 3:
            raise ValueError("Observation locations must have 3 columns (lat, lon, r)")

        if robust not in (None, "huber", "bisquare"):
            raise ValueError(
                f"Unknown robust weighting: '{robust}', "
                "options are None, 'huber', 'bisquare'"
            )

        if self.has_cf and np.any(self.sec_cf_loc[:, 2] >= obs_loc[:, 2].max()):
            warnings.warn(
                "Some curl-free SECs are at or above all observation "
                "locations. The magnetic field of a curl-free SEC is "
                "identically zero at and below its current shell "
                "(Fukushima's theorem), so these amplitudes are not "
                "constrained by the observations and will be set to zero. "
                "Add observations above the current shell (e.g. satellite "
                "data) to fit curl-free systems.",
                stacklevel=2,
            )

        if obs_B.ndim == 2:
            # Just a single snapshot given, so expand the dimensionality
            obs_B = obs_B[np.newaxis, ...]

        # Assume unit standard error of all measurements
        if obs_std is None:
            obs_std = np.ones_like(obs_B)

        ntimes = len(obs_B)
        # Flatten the components to do the math with shape (ntimes, nvariables)
        obs_B_flat = obs_B.reshape(ntimes, -1)
        obs_std_flat = obs_std.reshape(ntimes, -1)

        # Calculate the transfer functions, using cached values if possible
        if not np.array_equal(obs_loc, self._obs_loc):
            self._T_obs_flat = self._calc_T(obs_loc).reshape(-1, self.nsec)
            self._obs_loc = obs_loc

        self._solve_amplitudes(obs_B_flat, obs_std_flat, epsilon, mode)

        if robust is not None:
            self._irls(
                obs_B_flat,
                obs_std_flat,
                epsilon,
                mode,
                robust,
                robust_maxiter,
                robust_tol,
            )
        return self

    def _solve_amplitudes(
        self,
        obs_B_flat: np.ndarray,
        obs_std_flat: np.ndarray,
        epsilon: float,
        mode: str,
    ) -> None:
        """Solve for the SEC amplitudes and variances of all time steps."""
        ntimes = len(obs_B_flat)
        # Store the fit sec_amps in the object
        self.sec_amps = np.empty((ntimes, self.nsec))
        self.sec_amps_var = np.empty((ntimes, self.nsec))

        # The SVD only depends on the uncertainty pattern, so group time
        # steps that share an identical pattern (e.g. recurring station
        # dropouts) and compute the SVD once per unique pattern, applying
        # it to all matching time steps at once. With uniform uncertainties
        # this reduces to a single SVD and one vectorized solve.
        # Rows are grouped by their exact byte pattern, which is linear in
        # ntimes (np.unique(axis=0) sorts rows and is much slower here).
        pattern_index = np.empty(ntimes, dtype=np.intp)
        patterns: list[np.ndarray] = []
        seen: dict[bytes, int] = {}
        for i, std_row in enumerate(obs_std_flat):
            idx = seen.setdefault(std_row.tobytes(), len(patterns))
            if idx == len(patterns):
                patterns.append(std_row)
            pattern_index[i] = idx

        # Keep the inversion operators around so predict() can propagate
        # the amplitude covariance to prediction variances
        VWU_patterns = []
        for i, pattern_std in enumerate(patterns):
            t_mask = pattern_index == i
            VWU = self._compute_VWU(self._T_obs_flat, pattern_std, epsilon, mode)
            self.sec_amps[t_mask] = (obs_B_flat[t_mask] / pattern_std) @ VWU.T

            # amps = VWU @ (B / std). With Var(B) = std**2 the scaled
            # observations have unit variance, so Var(amps) = sum(VWU**2).
            # Observations eliminated with infinite std have exactly zero
            # columns in VWU and drop out automatically.
            self.sec_amps_var[t_mask] = np.sum(VWU**2, axis=1)
            VWU_patterns.append(VWU)

        self._VWU_patterns = VWU_patterns
        self._pattern_index = pattern_index

    def _irls(
        self,
        obs_B_flat: np.ndarray,
        obs_std_flat: np.ndarray,
        epsilon: float,
        mode: str,
        robust: str,
        maxiter: int,
        tol: float,
    ) -> None:
        """Refine the amplitudes with iteratively reweighted least squares.

        Standardized residuals are scaled by a per-time-step normalized
        median absolute deviation and mapped to weights with the Huber or
        Tukey bisquare functions (tuning constants chosen for 95%
        asymptotic efficiency on Gaussian data). Downweighting is applied
        by inflating the effective standard error of each observation, so
        the amplitude variances and prediction variances automatically
        account for the reduced influence of the flagged observations.
        """
        # Tuning constants for 95% asymptotic efficiency on Gaussian data
        c = {"huber": 1.345, "bisquare": 4.685}[robust]
        finite = np.isfinite(obs_std_flat)

        prev_amps = self.sec_amps
        for _ in range(maxiter):
            pred = prev_amps @ self._T_obs_flat.T
            resid = np.where(
                finite,
                (obs_B_flat - pred) / np.where(finite, obs_std_flat, 1.0),
                np.nan,
            )

            # Normalized median absolute deviation about zero of each time
            # step. A zero scale means the fit is (numerically) perfect, in
            # which case there are no outliers to reject.
            with warnings.catch_warnings():
                # All-NaN rows (fully eliminated time steps) are fine
                warnings.simplefilter("ignore", category=RuntimeWarning)
                scale = 1.4826 * np.nanmedian(np.abs(resid), axis=1, keepdims=True)
            scale = np.where(scale > 0, scale, np.inf)

            u = np.abs(np.where(finite, resid, 0.0)) / scale
            if robust == "huber":
                with np.errstate(divide="ignore"):
                    weights = np.minimum(1.0, c / np.where(u > 0, u, np.inf))
            else:
                weights = np.where(u < c, (1 - (u / c) ** 2) ** 2, 0.0)

            with np.errstate(divide="ignore"):
                eff_std = obs_std_flat / weights

            self._solve_amplitudes(obs_B_flat, eff_std, epsilon, mode)

            amp_scale = np.max(np.abs(self.sec_amps), initial=0.0)
            amp_change = np.max(np.abs(self.sec_amps - prev_amps), initial=0.0)
            if amp_change <= tol * amp_scale:
                break
            prev_amps = self.sec_amps

    def fit_unit_currents(self) -> "SECS":
        """Set all SECs to a unit current amplitude."""
        self.sec_amps = np.ones((1, self.nsec))
        self.sec_amps_var = np.zeros((1, self.nsec))
        # No inversion took place, so there is no covariance to propagate
        self._VWU_patterns = None
        self._pattern_index = None

        return self

    def predict(
        self, pred_loc: np.ndarray, J: bool = False, return_var: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Calculate the predicted magnetic field or currents.

        After a set of observations has been fit to this system we can
        predict the magnetic fields or currents at any other location. This
        function uses those fit amplitudes to predict at the requested locations.

        Parameters
        ----------
        pred_loc: ndarray (npred, 3 [lat, lon, r])
            An array containing the locations where the predictions are desired.

        J: boolean
            Whether to predict currents (J=True) or magnetic fields (J=False)
            Default: False (magnetic field prediction)

        return_var: boolean
            Whether to also return the variance of the predictions,
            propagated from the full posterior covariance of the fitted
            amplitudes (treating the obs_std passed to fit() as 1-sigma
            standard errors of independent observations).
            Default: False

        Returns
        -------
        ndarray (ntimes, npred, 3 [lat, lon, r])
            The predicted values calculated from the current amplitudes that were
            fit to this system. If return_var is True, a tuple of
            (predictions, variances) with identical shapes is returned.
        """
        if pred_loc.shape[-1] != 3:
            raise ValueError("Prediction locations must have 3 columns (lat, lon, r)")

        if len(self.sec_amps) == 0:
            raise ValueError(
                "There are no currents associated with the SECs, you need"
                "to call .fit() first to fit to some observations."
            )

        # T_pred shape: (npred, 3, nsec)
        # sec_amps shape: (ntimes, nsec)
        if J:
            # Predicting currents
            if not np.array_equal(pred_loc, self._pred_loc_J):
                self._T_pred_J = self._calc_J(pred_loc)
                self._pred_loc_J = pred_loc
            T_pred = self._T_pred_J
        else:
            # Predicting magnetic fields
            if not np.array_equal(pred_loc, self._pred_loc_B):
                self._T_pred_B = self._calc_T(pred_loc)
                self._pred_loc_B = pred_loc
            T_pred = self._T_pred_B

        pred = np.squeeze(np.tensordot(self.sec_amps, T_pred, (1, 2)))
        if not return_var:
            return pred

        if self._VWU_patterns is None or self._pattern_index is None:
            raise ValueError(
                "Prediction variances require amplitudes fit with fit(). "
                "Call fit() before predicting with return_var=True."
            )

        # pred = T_pred @ amps and amps = VWU @ (B / std) with the scaled
        # observations having unit variance, so
        # Cov(pred) = (T_pred @ VWU) @ (T_pred @ VWU).T and the variances
        # are the row sums of squares of M = T_pred @ VWU. The variance
        # only depends on the uncertainty pattern, so it is computed once
        # per pattern from the operators saved by fit().
        npred = len(pred_loc)
        T_pred_flat = T_pred.reshape(-1, self.nsec)
        var = np.empty((len(self.sec_amps), npred, 3))
        for i, VWU in enumerate(self._VWU_patterns):
            M = T_pred_flat @ VWU
            var[self._pattern_index == i] = np.sum(M**2, axis=1).reshape(npred, 3)

        return pred, np.squeeze(var)

    def predict_B(
        self, pred_loc: np.ndarray, return_var: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Calculate the predicted magnetic fields.

        After a set of observations has been fit to this system we can
        predict the magnetic fields or currents at any other location. This
        function uses those fit amplitudes to predict at the requested locations.

        Parameters
        ----------
        pred_loc: ndarray (npred, 3 [lat, lon, r])
            An array containing the locations where the predictions are desired.

        return_var: boolean
            Whether to also return the variance of the predictions.
            Default: False

        Returns
        -------
        ndarray (ntimes, npred, 3 [lat, lon, r])
            The predicted values calculated from the current amplitudes that were
            fit to this system. If return_var is True, a tuple of
            (predictions, variances) with identical shapes is returned.
        """
        return self.predict(pred_loc, return_var=return_var)

    def predict_J(
        self, pred_loc: np.ndarray, return_var: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Calculate the predicted currents.

        After a set of observations has been fit to this system we can
        predict the magnetic fields or currents at any other location. This
        function uses those fit amplitudes to predict at the requested locations.

        Parameters
        ----------
        pred_loc: ndarray (npred, 3 [lat, lon, r])
            An array containing the locations where the predictions are desired.

        return_var: boolean
            Whether to also return the variance of the predictions.
            Default: False

        Returns
        -------
        ndarray (ntimes, npred, 3 [lat, lon, r])
            The predicted values calculated from the current amplitudes that were
            fit to this system. If return_var is True, a tuple of
            (predictions, variances) with identical shapes is returned.
        """
        return self.predict(pred_loc, J=True, return_var=return_var)

    def _calc_T(self, obs_loc: np.ndarray) -> np.ndarray:
        """Calculate the T transfer matrix.

        The magnetic field transfer matrix to go from SEC locations to observation
        locations. It assumes unit current amplitudes that will then be
        scaled with the proper amplitudes later.
        """
        if self.has_df:
            T = T_df(obs_loc=obs_loc, sec_loc=self.sec_df_loc)

        if self.has_cf:
            T1 = T_cf(obs_loc=obs_loc, sec_loc=self.sec_cf_loc)
            # df is already present in T
            if self.has_df:
                T = np.concatenate([T, T1], axis=2)
            else:
                T = T1

        return T

    def _calc_J(self, obs_loc: np.ndarray) -> np.ndarray:
        """Calculate the J transfer matrix.

        The current transfer matrix to go from SEC locations to observation
        locations. It assumes unit current amplitudes that will then be
        scaled with the proper amplitudes later.
        """
        if self.has_df:
            J = J_df(obs_loc=obs_loc, sec_loc=self.sec_df_loc)

        if self.has_cf:
            J1 = J_cf(obs_loc=obs_loc, sec_loc=self.sec_cf_loc)
            # df is already present in T
            if self.has_df:
                J = np.concatenate([J, J1], axis=2)
            else:
                J = J1

        return J


def T_df(obs_loc: np.ndarray, sec_loc: np.ndarray) -> np.ndarray:
    """Calculate the divergence free magnetic field transfer function.

    The transfer function goes from SEC location to observation location
    and assumes unit current SECs at the given locations.

    Parameters
    ----------
    obs_loc : ndarray (nobs, 3 [lat, lon, r])
        The locations of the observation points.

    sec_loc : ndarray (nsec, 3 [lat, lon, r])
        The locations of the SEC points.

    Returns
    -------
    ndarray (nobs, 3, nsec)
        The T transfer matrix.
    """
    nobs = len(obs_loc)
    nsec = len(sec_loc)

    theta, alpha = _calc_angular_distance_and_bearing(obs_loc[:, :2], sec_loc[:, :2])

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    Br = np.empty((nobs, nsec))
    Btheta = np.empty((nobs, nsec))

    # Over locations: obs_r > sec_r
    over_locs = obs_loc[:, 2][:, np.newaxis] > sec_loc[:, 2][np.newaxis, :]
    if np.any(over_locs):
        # We use np.where because we are broadcasting 1d arrays
        # over_locs is a 2d array of booleans
        over_indices = np.where(over_locs)
        obs_r = obs_loc[over_indices[0], 2]
        sec_r = sec_loc[over_indices[1], 2]
        Br[over_locs], Btheta[over_locs] = _calc_T_df_over(
            obs_r, sec_r, cos_theta[over_locs]
        )

    # Under locations: obs_r <= sec_r
    under_locs = ~over_locs
    if np.any(under_locs):
        # We use np.where because we are broadcasting 1d arrays
        # over_locs is a 2d array of booleans
        under_indices = np.where(under_locs)
        obs_r = obs_loc[under_indices[0], 2]
        sec_r = sec_loc[under_indices[1], 2]
        Br[under_locs], Btheta[under_locs] = _calc_T_df_under(
            obs_r, sec_r, cos_theta[under_locs]
        )

    # If sin(theta) == 0: Btheta = 0
    # There is a possible 0/0 in the expansion when sec_loc == obs_loc
    Btheta = np.divide(
        Btheta, sin_theta, out=np.zeros_like(sin_theta), where=sin_theta != 0
    )

    # Transform back to Bx, By, Bz at each local point
    T = np.empty((nobs, 3, nsec))
    # alpha == angle (from cartesian x-axis (By), going towards y-axis (Bx))
    T[:, 0, :] = -Btheta * np.sin(alpha)
    T[:, 1, :] = -Btheta * np.cos(alpha)
    T[:, 2, :] = -Br

    return T


def _calc_T_df_under(
    obs_r: np.ndarray, sec_r: np.ndarray, cos_theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """T matrix for over locations (obs_r <= sec_r)."""
    mu0_over_4pi = 1e-7
    x = obs_r / sec_r
    factor = 1.0 / np.sqrt(1 - 2 * x * cos_theta + x**2)

    # Amm & Viljanen: Equation 9
    Br = mu0_over_4pi / obs_r * (factor - 1)

    # Amm & Viljanen: Equation 10
    Btheta = -mu0_over_4pi / obs_r * (factor * (x - cos_theta) + cos_theta)

    return Br, Btheta


def _calc_T_df_over(
    obs_r: np.ndarray, sec_r: np.ndarray, cos_theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """T matrix for over locations (obs_r > sec_r)."""
    mu0_over_4pi = 1e-7
    x = sec_r / obs_r
    factor = 1.0 / np.sqrt(1 - 2 * x * cos_theta + x**2)

    # Amm & Viljanen: Equation A.7
    Br = mu0_over_4pi * x / obs_r * (factor - 1)

    # Amm & Viljanen: Equation A.8
    Btheta = (
        -mu0_over_4pi
        / obs_r
        * (
            (obs_r - sec_r * cos_theta)
            / np.sqrt(obs_r**2 - 2 * obs_r * sec_r * cos_theta + sec_r**2)
            - 1
        )
    )

    return Br, Btheta


def T_cf(obs_loc: np.ndarray, sec_loc: np.ndarray) -> np.ndarray:
    """Calculate the curl free magnetic field transfer function.

    The transfer function goes from SEC location to observation location
    and assumes unit current SECs at the given locations.

    The curl-free SEC, together with its associated radial field-aligned
    currents (FACs), produces no magnetic field at or below its current
    shell (Fukushima's theorem generalized to a sphere) and a purely
    azimuthal magnetic field above it. A positive scaling factor
    corresponds to a FAC flowing into the ionosphere at the SEC pole,
    horizontal sheet currents directed away from the pole, and uniformly
    distributed FACs flowing out of the ionosphere elsewhere.

    Parameters
    ----------
    obs_loc : ndarray (nobs, 3 [lat, lon, r])
        The locations of the observation points.

    sec_loc : ndarray (nsec, 3 [lat, lon, r])
        The locations of the SEC points.

    Returns
    -------
    ndarray (nobs, 3, nsec)
        The T transfer matrix.
    """
    nobs = len(obs_loc)
    nsec = len(sec_loc)

    obs_r = obs_loc[:, 2][:, np.newaxis]
    sec_r = sec_loc[:, 2][np.newaxis, :]

    theta, alpha = _calc_angular_distance_and_bearing(obs_loc[:, :2], sec_loc[:, :2])

    mu0_over_4pi = 1e-7

    # Vanhamäki & Juusola: Equation 2.15
    # B_phi = -mu0 I0 / (4 pi r) cot(theta / 2) above the current shell
    tan_theta2 = np.tan(theta / 2.0)
    B_phi = np.divide(
        1.0,
        tan_theta2,
        out=np.ones_like(tan_theta2) * np.inf,
        where=tan_theta2 != 0.0,
    )
    B_phi *= -mu0_over_4pi / obs_r

    # Amm & Viljanen: Section 2 / Vanhamäki & Juusola: Equation 2.15
    # No magnetic field at or below the current shell (Fukushima's theorem)
    B_phi = np.where(obs_r > sec_r, B_phi, 0.0)

    # Transform back to Bx, By, Bz at each local point
    T = np.empty((nobs, 3, nsec))
    # alpha == angle (from cartesian x-axis (By), going towards y-axis (Bx))
    T[:, 0, :] = -B_phi * np.cos(alpha)
    T[:, 1, :] = B_phi * np.sin(alpha)
    T[:, 2, :] = 0.0

    return T


def J_df(obs_loc: np.ndarray, sec_loc: np.ndarray) -> np.ndarray:
    """Calculate the divergence free current density transfer function.

    The transfer function goes from SEC location to observation location
    and assumes unit current SECs at the given locations.

    Parameters
    ----------
    obs_loc : ndarray (nobs, 3 [lat, lon, r])
        The locations of the observation points.

    sec_loc : ndarray (nsec, 3 [lat, lon, r])
        The locations of the SEC points.

    Returns
    -------
    ndarray (nobs, 3, nsec)
        The J transfer matrix.
    """
    nobs = len(obs_loc)
    nsec = len(sec_loc)

    obs_r = obs_loc[:, 2][:, np.newaxis]
    sec_r = sec_loc[:, 2][np.newaxis, :]

    # Input to the distance calculations is degrees, output is in radians
    theta, alpha = _calc_angular_distance_and_bearing(obs_loc[:, :2], sec_loc[:, :2])

    # Amm & Viljanen: Equation 6
    tan_theta2 = np.tan(theta / 2.0)

    J_phi = 1.0 / (4 * np.pi * sec_r)
    J_phi = np.divide(
        J_phi,
        tan_theta2,
        out=np.ones_like(tan_theta2) * np.inf,
        where=tan_theta2 != 0.0,
    )
    # Only valid on the SEC shell
    J_phi[sec_r != obs_r] = 0.0

    # Transform back to Bx, By, Bz at each local point
    J = np.empty((nobs, 3, nsec))
    # alpha == angle (from cartesian x-axis (By), going towards y-axis (Bx))
    J[:, 0, :] = -J_phi * np.cos(alpha)
    J[:, 1, :] = J_phi * np.sin(alpha)
    J[:, 2, :] = 0.0

    return J


def J_cf(obs_loc: np.ndarray, sec_loc: np.ndarray) -> np.ndarray:
    """Calculate the curl free current density transfer function.

    The transfer function goes from SEC location to observation location
    and assumes unit current SECs at the given locations.

    A positive scaling factor corresponds to a field-aligned current (FAC)
    flowing into the ionosphere at the SEC pole, horizontal sheet currents
    directed away from the pole, and uniformly distributed FACs flowing out
    of the ionosphere on the rest of the shell so that the net FAC over the
    whole shell is zero (Amm & Viljanen, Fig. 1). Currents are only
    returned on the SEC shell itself; the radial components represent the
    FACs attached to the shell at that point.

    Parameters
    ----------
    obs_loc : ndarray (nobs, 3 [lat, lon, r])
        The locations of the observation points.

    sec_loc : ndarray (nsec, 3 [lat, lon, r])
        The locations of the SEC points.

    Returns
    -------
    ndarray (nobs, 3, nsec)
        The J transfer matrix.
    """
    nobs = len(obs_loc)
    nsec = len(sec_loc)

    obs_r = obs_loc[:, 2][:, np.newaxis]
    sec_r = sec_loc[:, 2][np.newaxis, :]

    theta, alpha = _calc_angular_distance_and_bearing(obs_loc[:, :2], sec_loc[:, :2])

    # Amm & Viljanen: Equation 7
    tan_theta2 = np.tan(theta / 2.0)

    J_theta = 1.0 / (4 * np.pi * sec_r)
    J_theta = np.divide(
        J_theta,
        tan_theta2,
        out=np.ones_like(tan_theta2) * np.inf,
        where=tan_theta2 != 0,
    )
    # Uniformly distributed FACs flowing out of the ionosphere (upward)
    # around the globe return the line current flowing into the ionosphere
    # at the pole, so the net FAC integrated over the globe is zero.
    # The pole entry is a finite marker for the delta-function line current
    # flowing into the ionosphere (downward).
    J_r = np.ones(J_theta.shape) / (4 * np.pi * sec_r**2)
    J_r[theta == 0.0] = -1.0

    # Only valid on the SEC shell
    J_theta[sec_r != obs_r] = 0.0
    J_r[sec_r != obs_r] = 0.0

    # Transform back to Bx, By, Bz at each local point
    J = np.empty((nobs, 3, nsec))
    # alpha == angle (from cartesian x-axis (By), going towards y-axis (Bx))

    J[:, 0, :] = -J_theta * np.sin(alpha)
    J[:, 1, :] = -J_theta * np.cos(alpha)
    J[:, 2, :] = -J_r

    return J


def _calc_angular_distance_and_bearing(
    latlon1: np.ndarray, latlon2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the angular distance and bearing between two points.

    This function calculates the angular distance in radians
    between two latitude and longitude points. It also calculates
    the bearing (direction) from point 1 to point 2 going from the
    cartesian x-axis towards the cartesian y-axis.

    Parameters
    ----------
    latlon1 : ndarray of shape (N, 2)
        Array of N (latitude, longitude) points.
    latlon2 : ndarray of shape (M, 2)
        Array of M (latitude, longitude) points.

    Returns
    -------
    theta : ndarray of shape (N, M)
        Angular distance in radians between each point in latlon1 and latlon2.
    alpha : ndarray of shape (N, M)
        Bearing from each point in latlon1 to each point in latlon2.
    """
    latlon1_rad = np.deg2rad(latlon1)
    latlon2_rad = np.deg2rad(latlon2)

    lat1 = latlon1_rad[:, 0][:, None]
    lon1 = latlon1_rad[:, 1][:, None]
    lat2 = latlon2_rad[:, 0][None, :]
    lon2 = latlon2_rad[:, 1][None, :]

    cos_lat1 = np.cos(lat1)
    sin_lat1 = np.sin(lat1)
    cos_lat2 = np.cos(lat2)
    sin_lat2 = np.sin(lat2)

    dlon = lon2 - lon1
    cos_dlon = np.cos(dlon)
    sin_dlon = np.sin(dlon)

    # x, y, and dot below are the components of the Vincenty formula for
    # angular distance. Unlike the law-of-cosines + arccos form (which this
    # replaced), it has no restricted input domain, so it can't return NaN
    # for coincident points due to floating-point round-off, and it stays
    # accurate when the two points are close together, where arccos loses
    # precision rapidly. x and y also double as the bearing components below.
    x = cos_lat2 * sin_dlon
    y = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_dlon
    dot = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_dlon

    # theta == angular distance between two points
    theta = np.arctan2(np.hypot(x, y), dot)

    # alpha == bearing, going from point1 to point2
    #          angle (from cartesian x-axis (By), going towards y-axis (Bx))
    # Used to rotate the SEC coordinate frame into the observation coordinate
    # frame.
    # SEC coordinates are: theta (colatitude (+ away from North Pole)),
    #                      phi (longitude, + east), r (+ out)
    # Obs coordinates are: X (+ north), Y (+ east), Z (+ down)
    alpha = np.pi / 2 - np.arctan2(x, y)

    return theta, alpha


def calc_angular_distance(latlon1: np.ndarray, latlon2: np.ndarray) -> np.ndarray:
    """Calculate the angular distance between a set of points.

    This function calculates the angular distance in radians
    between any number of latitude and longitude points.

    Parameters
    ----------
    latlon1 : ndarray (n, 2 [lat, lon])
        An array of n (latitude, longitude) points.

    latlon2 : ndarray (m, 2 [lat, lon])
        An array of m (latitude, longitude) points.

    Returns
    -------
    ndarray (n, m)
        The array of distances between the input arrays.
    """
    lat1 = np.deg2rad(latlon1[:, 0])[:, np.newaxis]
    lon1 = np.deg2rad(latlon1[:, 1])[:, np.newaxis]
    lat2 = np.deg2rad(latlon2[:, 0])[np.newaxis, :]
    lon2 = np.deg2rad(latlon2[:, 1])[np.newaxis, :]

    cos_lat1 = np.cos(lat1)
    sin_lat1 = np.sin(lat1)
    cos_lat2 = np.cos(lat2)
    sin_lat2 = np.sin(lat2)

    dlon = lon2 - lon1
    cos_dlon = np.cos(dlon)
    sin_dlon = np.sin(dlon)

    # Vincenty formula: unlike the law-of-cosines + arccos form (which this
    # replaced), it has no restricted input domain, so it can't return NaN
    # for coincident points due to floating-point round-off, and it stays
    # accurate when the two points are close together, where arccos loses
    # precision rapidly.
    x = cos_lat2 * sin_dlon
    y = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_dlon
    dot = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_dlon

    # theta == angular distance between two points
    theta = np.arctan2(np.hypot(x, y), dot)
    return theta


def calc_bearing(latlon1: np.ndarray, latlon2: np.ndarray) -> np.ndarray:
    """Calculate the bearing (direction) between a set of points.

    This function calculates the bearing in radians
    between any number of latitude and longitude points.
    It is the direction from point 1 to point 2 going from the
    cartesian x-axis towards the cartesian y-axis.

    Parameters
    ----------
    latlon1 : ndarray (n, 2 [lat, lon])
        An array of n (latitude, longitude) points.

    latlon2 : ndarray (m, 2 [lat, lon])
        An array of m (latitude, longitude) points.

    Returns
    -------
    ndarray (n, m)
        The array of bearings between the input arrays.
    """
    lat1 = np.deg2rad(latlon1[:, 0])[:, np.newaxis]
    lon1 = np.deg2rad(latlon1[:, 1])[:, np.newaxis]
    lat2 = np.deg2rad(latlon2[:, 0])[np.newaxis, :]
    lon2 = np.deg2rad(latlon2[:, 1])[np.newaxis, :]

    dlon = lon2 - lon1

    # alpha == bearing, going from point1 to point2
    #          angle (from cartesian x-axis (By), going towards y-axis (Bx))
    # Used to rotate the SEC coordinate frame into the observation coordinate
    # frame.
    # SEC coordinates are: theta (colatitude (+ away from North Pole)),
    #                      phi (longitude, + east), r (+ out)
    # Obs coordinates are: X (+ north), Y (+ east), Z (+ down)
    alpha = np.pi / 2 - np.arctan2(
        np.sin(dlon) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
    )
    return alpha
