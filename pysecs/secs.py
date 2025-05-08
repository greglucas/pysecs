"""Spherical Elementary Current System (SECS) module.

Calculate magnetic field transfer functions and fit a system (SECS) to observations.
"""

import numpy as np


class SECS:
    """Spherical Elementary Current System (SECS).

    The algorithm is implemented directly in spherical coordinates
    from the equations of the 1999 Amm & Viljanen paper [1]_.

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

        # Keep some values around for cache lookups
        self._obs_loc = None
        self._T_obs_flat = None
        self._pred_loc_B = None
        self._T_pred_B = None
        self._pred_loc_J = None
        self._T_pred_J = None

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

        # Truncate and build VWU
        U = U[:, valid]
        S = S[valid]
        Vh = Vh[valid, :]
        W = 1.0 / S

        return Vh.T @ (np.diag(W) @ U.T)

    def fit(
        self,
        obs_loc: np.ndarray,
        obs_B: np.ndarray,
        obs_std: np.ndarray | None = None,
        epsilon: float = 0.05,
        mode: str = "relative",
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

        obs_std : ndarray (ntimes, nobs, 3 [varX, varY, varZ]), optional
            Standard error of vector components at each observation location.
            This can be used to weight different observations more/less heavily.
            An infinite value eliminates the observation from the fit.
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
        """
        if obs_loc.shape[-1] != 3:
            raise ValueError("Observation locations must have 3 columns (lat, lon, r)")

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

        # Store the fit sec_amps in the object
        self.sec_amps = np.empty((ntimes, self.nsec))
        self.sec_amps_var = np.empty((ntimes, self.nsec))

        if np.allclose(obs_std_flat, obs_std_flat[0]):
            # The SVD is the same for all time steps, so we can calculate it once
            # and broadcast it to all time steps avoiding the for-loop below
            VWU = self._compute_VWU(self._T_obs_flat, obs_std_flat[0], epsilon, mode)
            self.sec_amps[:] = (obs_B_flat / obs_std_flat) @ VWU.T

            valid = np.isfinite(obs_std_flat[0])
            VWU_masked = VWU[:, valid]
            std_masked = obs_std_flat[0, valid]
            self.sec_amps_var[:] = np.sum((VWU_masked * std_masked) ** 2, axis=1)
        else:
            prev_std = None
            VWU = None
            for i in range(ntimes):
                if prev_std is None or not np.allclose(
                    obs_std_flat[i], prev_std, atol=1e-12, rtol=1e-12
                ):
                    VWU = self._compute_VWU(
                        self._T_obs_flat, obs_std_flat[i], epsilon, mode
                    )
                    prev_std = obs_std_flat[i]

                self.sec_amps[i] = VWU @ (obs_B_flat[i] / obs_std_flat[i])

                valid = np.isfinite(obs_std_flat[i])
                VWU_masked = VWU[:, valid]
                std_masked = obs_std_flat[i, valid]
                self.sec_amps_var[i] = np.sum((VWU_masked * std_masked) ** 2, axis=1)
        return self

    def fit_unit_currents(self) -> "SECS":
        """Set all SECs to a unit current amplitude."""
        self.sec_amps = np.ones((1, self.nsec))

        return self

    def predict(self, pred_loc: np.ndarray, J: bool = False) -> np.ndarray:
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

        Returns
        -------
        ndarray (ntimes, npred, 3 [lat, lon, r])
            The predicted values calculated from the current amplitudes that were
            fit to this system.
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

        return np.squeeze(np.tensordot(self.sec_amps, T_pred, (1, 2)))

    def predict_B(self, pred_loc: np.ndarray) -> np.ndarray:
        """Calculate the predicted magnetic fields.

        After a set of observations has been fit to this system we can
        predict the magnetic fields or currents at any other location. This
        function uses those fit amplitudes to predict at the requested locations.

        Parameters
        ----------
        pred_loc: ndarray (npred, 3 [lat, lon, r])
            An array containing the locations where the predictions are desired.

        Returns
        -------
        ndarray (ntimes, npred, 3 [lat, lon, r])
            The predicted values calculated from the current amplitudes that were
            fit to this system.
        """
        return self.predict(pred_loc)

    def predict_J(self, pred_loc: np.ndarray) -> np.ndarray:
        """Calculate the predicted currents.

        After a set of observations has been fit to this system we can
        predict the magnetic fields or currents at any other location. This
        function uses those fit amplitudes to predict at the requested locations.

        Parameters
        ----------
        pred_loc: ndarray (npred, 3 [lat, lon, r])
            An array containing the locations where the predictions are desired.

        Returns
        -------
        ndarray (ntimes, npred, 3 [lat, lon, r])
            The predicted values calculated from the current amplitudes that were
            fit to this system.
        """
        return self.predict(pred_loc, J=True)

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

    obs_r = obs_loc[:, 2][:, np.newaxis]
    sec_r = sec_loc[:, 2][np.newaxis, :]

    theta, alpha = _calc_angular_distance_and_bearing(obs_loc[:, :2], sec_loc[:, :2])

    # magnetic permeability
    mu0 = 4 * np.pi * 1e-7

    # simplify calculations by storing this ratio
    x = obs_r / sec_r

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    factor = 1.0 / np.sqrt(1 - 2 * x * cos_theta + x**2)

    # Amm & Viljanen: Equation 9
    Br = mu0 / (4 * np.pi * obs_r) * (factor - 1)

    # Amm & Viljanen: Equation 10 (transformed to try and eliminate trig operations and
    #                              divide by zeros)
    Btheta = -mu0 / (4 * np.pi * obs_r) * (factor * (x - cos_theta) + cos_theta)
    # If sin(theta) == 0: Btheta = 0
    # There is a possible 0/0 in the expansion when sec_loc == obs_loc
    Btheta = np.divide(
        Btheta, sin_theta, out=np.zeros_like(sin_theta), where=sin_theta != 0
    )

    # When observation points radii are outside of the sec locations
    under_locs = sec_r < obs_r

    # NOTE: If any SECs are below observations the math will be done on all points.
    #       This could be updated to only work on the locations where this condition
    #       occurs, but would make the code messier, with minimal performance gain
    #       except for very large matrices.
    if np.any(under_locs):
        # Flipped from previous case
        x = sec_r / obs_r

        # Amm & Viljanen: Equation A.7
        Br2 = (
            mu0
            * x
            / (4 * np.pi * obs_r)
            * (1.0 / np.sqrt(1 - 2 * x * cos_theta + x**2) - 1)
        )

        # Amm & Viljanen: Equation A.8
        Btheta2 = (
            -mu0
            / (4 * np.pi * obs_r)
            * (
                (obs_r - sec_r * cos_theta)
                / np.sqrt(obs_r**2 - 2 * obs_r * sec_r * cos_theta + sec_r**2)
                - 1
            )
        )
        Btheta2 = np.divide(
            Btheta2, sin_theta, out=np.zeros_like(sin_theta), where=sin_theta != 0
        )

        # Update only the locations where secs are under observations
        Btheta[under_locs] = Btheta2[under_locs]
        Br[under_locs] = Br2[under_locs]

    # Transform back to Bx, By, Bz at each local point
    T = np.empty((nobs, 3, nsec))
    # alpha == angle (from cartesian x-axis (By), going towards y-axis (Bx))
    T[:, 0, :] = -Btheta * np.sin(alpha)
    T[:, 1, :] = -Btheta * np.cos(alpha)
    T[:, 2, :] = -Br

    return T


def T_cf(obs_loc: np.ndarray, sec_loc: np.ndarray) -> np.ndarray:
    """Calculate the curl free magnetic field transfer function.

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
    raise NotImplementedError(
        "Curl Free Magnetic Field Transfers are not implemented yet."
    )


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
    """Calculate the curl free magnetic field transfer function.

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
    # Uniformly directed FACs around the globe, except the pole
    # Integrated over the globe, this will lead to zero
    J_r = -np.ones(J_theta.shape) / (4 * np.pi * sec_r**2)
    J_r[theta == 0.0] = 1.0

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

    # theta == angular distance between two points
    theta = np.arccos(sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_dlon)

    # alpha == bearing, going from point1 to point2
    #          angle (from cartesian x-axis (By), going towards y-axis (Bx))
    # Used to rotate the SEC coordinate frame into the observation coordinate
    # frame.
    # SEC coordinates are: theta (colatitude (+ away from North Pole)),
    #                      phi (longitude, + east), r (+ out)
    # Obs coordinates are: X (+ north), Y (+ east), Z (+ down)
    x = cos_lat2 * sin_dlon
    y = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_dlon
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

    dlon = lon2 - lon1

    # theta == angular distance between two points
    theta = np.arccos(
        np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon)
    )
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
