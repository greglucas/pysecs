import numpy as np


class SECS:
    """Spherical elementary current system (SECS).

    Algorithm implemented directly in spherical coordinates
    from the equations of the 1999 Amm & Viljanen paper.

    Amm, O., and A. Viljanen. "Ionospheric disturbance magnetic field continuation
    from the ground to the ionosphere using spherical elementary current systems."
    Earth, Planets and Space 51.6 (1999): 431-440. doi:10.1186/BF03352247

    Parameters
    ----------

    sec_df_loc : array (nsec x 3 [lat, lon, r])
        Contains latitude, longiutde, and radius of the divergence free (df) SEC locations

    sec_cf_loc : array (nsec x 3 [lat, lon, r])
        Contains latitude, longiutde, and radius of the curl free (cf) SEC locations


    """

    def __init__(self, sec_df_loc=None, sec_cf_loc=None):

        if sec_df_loc is None and sec_cf_loc is None:
            raise ValueError("Must initialize the object with SEC locations")

        self.sec_df_loc = sec_df_loc
        self.sec_cf_loc = sec_cf_loc

        if self.sec_df_loc is not None:
            self.sec_df_loc = np.asarray(sec_df_loc)
            if self.sec_df_loc.shape[-1] != 3:
                raise ValueError("SEC DF locations must have 3 columns (lat, lon, r)")
            if self.sec_df_loc.ndim == 1:
                # Add an empty dimension if only one SEC location is passed in
                self.sec_df_loc = self.sec_df_loc[np.newaxis, ...]

        if self.sec_cf_loc is not None:
            self.sec_cf_loc = np.asarray(sec_cf_loc)
            if self.sec_cf_loc.shape[-1] != 3:
                raise ValueError("SEC CF locations must have 3 columns (lat, lon, r)")
            if self.sec_cf_loc.ndim == 1:
                # Add an empty dimension if only one SEC location is passed in
                self.sec_cf_loc = self.sec_cf_loc[np.newaxis, ...]

        # Storage of the scaling factors
        self.sec_amps = None
        self.sec_amps_var = None

    @property
    def has_df(self):
        return self.sec_df_loc is not None

    @property
    def has_cf(self):
        return self.sec_cf_loc is not None

    @property
    def nsec(self):
        nsec = 0
        if self.has_df:
            nsec += len(self.sec_df_loc)
        if self.has_cf:
            nsec += len(self.sec_cf_loc)
        return nsec

    def fit(self, obs_loc, obs_B, obs_var=None, epsilon=0.05):
        """Fits the SECS to the given observations.

        obs_locs : array (nobs x 3 [lat, lon, r])
            Contains latitude, longitude, and radius of the observation locations
            (place where the measurements are made)

        obs_B: array (ntimes x nobs x 3 [Bx, By, Bz])
            An array containing the measured/observed B-fields.

        obs_var : array (nobs x 3 [varX, varY, varZ])
            Variances in the components at each observation location. Can be used to
            weight different observation locations more/less heavily. Infinite variance
            effectively eliminates the observation from the fit.
            Default: ones(nobs x 3) equal weights

        epsilon : float
            Value used to regularize/smooth the SECS amplitudes. Multiplied by the
            largest singular value obtained from SVD.
            Default: 0.05
        """
        if obs_loc.shape[-1] != 3:
            raise ValueError("Observation locations must have 3 columns (lat, lon, r)")

        # Assume unit variance of all measurements
        if obs_var is None:
            obs_var = np.ones(obs_loc.shape)

        if obs_B.ndim == 2:
            # Just a single snapshot given, so expand the dimensionality
            obs_B = obs_B[np.newaxis, ...]

        ntimes = len(obs_B)

        # Calculate the transfer functions
        T_obs = self._calc_T(obs_loc)

        # Calculate the singular value decomposition (SVD)

        # T has shape: (nobs, 3, nsec),
        # flatten the directions and reshape it to (nobs*3, nsec)
        U, S, Vh = np.linalg.svd(T_obs.reshape(-1, self.nsec) /
                                 obs_var.ravel()[:, np.newaxis], full_matrices=False)

        # Divide by infinity (1/S) gives zero weights
        W = 1./S
        # Eliminate the small singular values (less than epsilon)
        # by giving them zero weight
        W[S < epsilon*S.max()] = 0.

        VWU = np.dot(Vh.T, np.dot(np.diag(W), U.T))

        # Store the fit 'sec_amps' in the object
        # shape: ntimes x nsec
        self.sec_amps = np.dot(VWU, (obs_B/obs_var).reshape(ntimes, -1).T).T
        # Maybe want the variance of the predictions sometime later...?
        # shape: nsec
        self.sec_amps_var = np.sum((Vh.T * W)**2, axis=1)

        return self

    def fit_unit_currents(self):
        """Sets all SECs to a unit current amplitude."""
        self.sec_amps = np.ones((1, self.nsec))

        return self

    def predict(self, pred_loc, J=False):
        """Returns an array of predictions, given unit SEC currents.

        pred_loc: array (npred x 3 [lat, lon, r])
            An array containing the locations where the predictions are desired.

        J: boolean (default: False)
            Whether to predict currents (J=True) or magnetic fields (J=False)

        """
        if pred_loc.shape[-1] != 3:
            raise ValueError("Prediction locations must have 3 columns (lat, lon, r)")

        if self.sec_amps is None:
            raise ValueError("There are currently no currents associated with the SECs," +
                             "you need to call .fit() first to fit to some observations.")

        # T_pred shape=(npred x 3 x nsec)
        # sec_amps shape=(nsec x ntimes)
        if J:
            # Predicting currents
            T_pred = self._calc_J(pred_loc)
        else:
            # Predicting magnetic fields
            T_pred = self._calc_T(pred_loc)

        # NOTE: dot product is slow on multi-dimensional arrays (i.e. > 2 dimensions)
        #       Therefore this is implemented as tensordot, and the arguments are
        #       arranged to eliminate needs of transposing things later.
        #       The dot product is done over the SEC locations, so the final output
        #       is of shape: (ntimes x npred x 3)

        return np.squeeze(np.tensordot(self.sec_amps, T_pred, (1, 2)))

    def predict_B(self, pred_loc):
        """Returns an array of magnetic field predictions.

        pred_loc: array (npred x 3 [lat, lon, r])
            An array containing the locations where the predictions are desired.

        """
        return self.predict(pred_loc)

    def predict_J(self, pred_loc):
        """Returns an array of current predictions.

        pred_loc: array (npred x 3 [lat, lon, r])
            An array containing the locations where the predictions are desired.

        """
        return self.predict(pred_loc, J=True)

    def _calc_T(self, obs_loc):
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

    def _calc_J(self, obs_loc):
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


def T_df(obs_loc, sec_loc):
    """Calculates the transfer function from SEC location to observation location.

    The transfer function assumes unit current SECs at the given locations.

    Parameters:
    -----------
    obs_loc : array (nobs, 3 [lat, lon, r])

    sec_loc : array (nsec, 3 [lat, lon, r])

    Returns:
    --------
    T : array (nobs, 3, nsec)

    """
    nobs = len(obs_loc)
    nsec = len(sec_loc)

    obs_r = obs_loc[:, 2][:, np.newaxis]
    sec_r = sec_loc[:, 2][np.newaxis, :]

    theta = calc_angular_distance(obs_loc[:, :2], sec_loc[:, :2])
    alpha = calc_bearing(obs_loc[:, :2], sec_loc[:, :2])

    # magnetic permeability
    mu0 = 4*np.pi*1e-7

    # simplify calculations by storing this ratio
    x = obs_r/sec_r

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    factor = 1./np.sqrt(1 - 2*x*cos_theta + x**2)

    # Amm & Viljanen: Equation 9
    Br = mu0/(4*np.pi*obs_r) * (factor - 1)

    # Amm & Viljanen: Equation 10 (transformed to try and eliminate trig operations and
    #                              divide by zeros)
    Btheta = -mu0/(4*np.pi*obs_r) * (factor*(x - cos_theta) + cos_theta)
    # If sin(theta) == 0: Btheta = 0
    # There is a possible 0/0 in the expansion when sec_loc == obs_loc
    Btheta = np.divide(Btheta, sin_theta, out=np.zeros_like(sin_theta),
                       where=sin_theta != 0)

    # When observation points radii are outside of the sec locations
    under_locs = sec_r < obs_r

    # NOTE: If any SECs are below observations the math will be done on all points.
    #       This could be updated to only work on the locations where this condition
    #       occurs, but would make the code messier, with minimal performance gain
    #       except for very large matrices.
    if np.any(under_locs):
        # Flipped from previous case
        x = sec_r/obs_r

        # Amm & Viljanen: Equation A.7
        Br2 = mu0*x/(4*np.pi*obs_r) * (1./np.sqrt(1 - 2*x*cos_theta + x**2) - 1)

        # Amm & Viljanen: Equation A.8
        Btheta2 = - mu0 / (4*np.pi*obs_r) * ((obs_r-sec_r*cos_theta) /
                                             np.sqrt(obs_r**2 -
                                                     2*obs_r*sec_r*cos_theta +
                                                     sec_r**2) - 1)
        Btheta2 = np.divide(Btheta2, sin_theta, out=np.zeros_like(sin_theta),
                            where=sin_theta != 0)

        # Update only the locations where secs are under observations
        Btheta[under_locs] = Btheta2[under_locs]
        Br[under_locs] = Br2[under_locs]

    # Transform back to Bx, By, Bz at each local point
    T = np.empty((nobs, 3, nsec))
    # alpha == angle (from cartesian x-axis (By), going towards y-axis (Bx))
    T[:, 0, :] = -Btheta*np.sin(alpha)
    T[:, 1, :] = -Btheta*np.cos(alpha)
    T[:, 2, :] = -Br

    return T


def T_cf(obs_loc, sec_loc):
    """Calculates the transfer function from SEC location to observation location.

    The transfer function assumes unit current SECs at the given locations.

    Parameters:
    -----------
    obs_loc : array (nobs, 3 [lat, lon, r])

    sec_loc : array (nsec, 3 [lat, lon, r])

    Returns:
    --------
    T : array (nobs, 3, nsec)

    """
    raise NotImplementedError("Curl Free Magnetic Field Transfers are not implemented yet.")


def J_df(obs_loc, sec_loc):
    """Calculates the transfer function from SEC location to observation location.

    The transfer function assumes unit current SECs at the given locations.

    Parameters:
    -----------
    obs_loc : array (nobs, 3 [lat, lon, r])

    sec_loc : array (nsec, 3 [lat, lon, r])

    Returns:
    --------
    J : array (nobs, 3, nsec)

    """
    nobs = len(obs_loc)
    nsec = len(sec_loc)

    obs_r = obs_loc[:, 2][:, np.newaxis]
    sec_r = sec_loc[:, 2][np.newaxis, :]

    # Input to the distance calculations is degrees, output is in radians
    theta = calc_angular_distance(obs_loc[:, :2], sec_loc[:, :2])
    alpha = calc_bearing(obs_loc[:, :2], sec_loc[:, :2])

    # Amm & Viljanen: Equation 6
    tan_theta2 = np.tan(theta/2.)

    J_phi = 1./(4*np.pi*sec_r)
    J_phi = np.divide(J_phi, tan_theta2, out=np.ones_like(tan_theta2)*np.inf,
                      where=tan_theta2 != 0.)
    # Only valid on the SEC shell
    J_phi[sec_r != obs_r] = 0.

    # Transform back to Bx, By, Bz at each local point
    J = np.empty((nobs, 3, nsec))
    # alpha == angle (from cartesian x-axis (By), going towards y-axis (Bx))
    J[:, 0, :] = -J_phi*np.cos(alpha)
    J[:, 1, :] = J_phi*np.sin(alpha)
    J[:, 2, :] = 0.

    return J


def J_cf(obs_loc, sec_loc):
    """Calculates the transfer function from SEC location to observation location.

    The transfer function assumes unit current SECs at the given locations.

    Parameters:
    -----------
    obs_loc : array (nobs, 3 [lat, lon, r])

    sec_loc : array (nsec, 3 [lat, lon, r])

    Returns:
    --------
    J : array (nobs, 3, nsec)

    """
    nobs = len(obs_loc)
    nsec = len(sec_loc)

    obs_r = obs_loc[:, 2][:, np.newaxis]
    sec_r = sec_loc[:, 2][np.newaxis, :]

    theta = calc_angular_distance(obs_loc[:, :2], sec_loc[:, :2])
    alpha = calc_bearing(obs_loc[:, :2], sec_loc[:, :2])

    # Amm & Viljanen: Equation 7
    tan_theta2 = np.tan(theta/2.)

    J_theta = 1./(4*np.pi*sec_r)
    J_theta = np.divide(J_theta, tan_theta2, out=np.ones_like(tan_theta2)*np.inf,
                        where=tan_theta2 != 0)
    # Uniformly directed FACs around the globe, except the pole
    # Integrated over the globe, this will lead to zero
    J_r = -np.ones(J_theta.shape)/(4*np.pi*sec_r**2)
    J_r[theta == 0.] = 1.

    # Only valid on the SEC shell
    J_theta[sec_r != obs_r] = 0.
    J_r[sec_r != obs_r] = 0.

    # Transform back to Bx, By, Bz at each local point
    J = np.empty((nobs, 3, nsec))
    # alpha == angle (from cartesian x-axis (By), going towards y-axis (Bx))

    J[:, 0, :] = -J_theta*np.sin(alpha)
    J[:, 1, :] = -J_theta*np.cos(alpha)
    J[:, 2, :] = -J_r

    return J


def calc_angular_distance(latlon1, latlon2):
    lat1 = np.deg2rad(latlon1[:, 0])[:, np.newaxis]
    lon1 = np.deg2rad(latlon1[:, 1])[:, np.newaxis]
    lat2 = np.deg2rad(latlon2[:, 0])[np.newaxis, :]
    lon2 = np.deg2rad(latlon2[:, 1])[np.newaxis, :]

    dlon = lon2 - lon1

    # theta == angular distance between two points
    theta = np.arccos(np.sin(lat1)*np.sin(lat2) +
                      np.cos(lat1)*np.cos(lat2)*np.cos(dlon))
    return theta


def calc_bearing(latlon1, latlon2):
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
    alpha = np.pi/2 - np.arctan2(np.sin(dlon)*np.cos(lat2),
                                 np.cos(lat1)*np.sin(lat2) -
                                 np.sin(lat1)*np.cos(lat2)*np.cos(dlon))
    return alpha
