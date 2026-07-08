"""Physics-invariant tests for the SECS transfer functions.

These tests verify Maxwell-equation consistency of the analytic transfer
functions rather than re-transcribing the formulas:

1. div B = 0 everywhere.
2. The discontinuity of B across the current shell reproduces the sheet
   currents of Amm & Viljanen (1999) Eqs. 6 and 7.
3. curl B = 0 in current-free regions (divergence-free system off-shell).
4. curl B of the curl-free system above the shell equals mu0 times the
   distributed field-aligned current density.

These are the four consistency checks suggested by Vanhamäki & Juusola
(2020, Sect. 2.4). Additional anchors: the exact 10 nT value from
Amm & Viljanen (1999) Fig. 2, the infinite-line-current limit of the
curl-free field, round-trip amplitude recovery, and rotation invariance.
"""

import numpy as np
from numpy.testing import assert_allclose

import pysecs


R_EARTH = 6371e3
MU0 = 4 * np.pi * 1e-7
SEC_SHELL = R_EARTH + 110e3
SEC_LATLONR = np.array([[47.0, 12.0, SEC_SHELL]])


def spherical_components(B_xyz):
    """Convert (Bx north, By east, Bz down) to (Br, Btheta_colat, Bphi)."""
    return -B_xyz[2], -B_xyz[0], B_xyz[1]


def field_at(func, sec_latlonr, lat, lon, r):
    """Evaluate a single-SEC transfer function at a single point."""
    T = func(np.atleast_2d([lat, lon, r]), sec_latlonr)
    return np.array([T[0, 0, 0], T[0, 1, 0], T[0, 2, 0]])


def numerical_div_curl(func, sec_latlonr, lat, lon, r, dang=1e-5, dr=25.0):
    """Numerical divergence and curl of B in spherical coordinates.

    Central differences in radius, colatitude, and longitude.
    Returns (div, [curl_r, curl_theta, curl_phi], |B|).
    """
    theta_c = np.deg2rad(90.0 - lat)
    dang_deg = np.rad2deg(dang)

    def sph(lat_, lon_, r_):
        return spherical_components(field_at(func, sec_latlonr, lat_, lon_, r_))

    # Radial derivatives
    Br_p, Bt_p, Bp_p = sph(lat, lon, r + dr)
    Br_m, Bt_m, Bp_m = sph(lat, lon, r - dr)
    dr_r2Br = ((r + dr) ** 2 * Br_p - (r - dr) ** 2 * Br_m) / (2 * dr)
    dr_rBp = ((r + dr) * Bp_p - (r - dr) * Bp_m) / (2 * dr)
    dr_rBt = ((r + dr) * Bt_p - (r - dr) * Bt_m) / (2 * dr)

    # Colatitude derivatives (colatitude increases as latitude decreases)
    Br_tp, Bt_tp, Bp_tp = sph(lat - dang_deg, lon, r)
    Br_tm, Bt_tm, Bp_tm = sph(lat + dang_deg, lon, r)
    st_p = np.sin(theta_c + dang)
    st_m = np.sin(theta_c - dang)
    dth_sinBt = (st_p * Bt_tp - st_m * Bt_tm) / (2 * dang)
    dth_sinBp = (st_p * Bp_tp - st_m * Bp_tm) / (2 * dang)
    dth_Br = (Br_tp - Br_tm) / (2 * dang)

    # Longitude derivatives
    Br_pp, Bt_pp, Bp_pp = sph(lat, lon + dang_deg, r)
    Br_pm, Bt_pm, Bp_pm = sph(lat, lon - dang_deg, r)
    dph_Br = (Br_pp - Br_pm) / (2 * dang)
    dph_Bt = (Bt_pp - Bt_pm) / (2 * dang)
    dph_Bp = (Bp_pp - Bp_pm) / (2 * dang)

    st = np.sin(theta_c)
    div = dr_r2Br / r**2 + dth_sinBt / (r * st) + dph_Bp / (r * st)
    curl_r = (dth_sinBp - dph_Bt) / (r * st)
    curl_t = dph_Br / (r * st) - dr_rBp / r
    curl_p = (dr_rBt - dth_Br) / r

    B0 = np.linalg.norm(sph(lat, lon, r))
    return div, np.array([curl_r, curl_t, curl_p]), B0


def test_df_divergence_free_below_and_above():
    "div B = 0 for the divergence-free system on both sides of the shell."
    for lat, lon, r in [
        (40.0, 10.0, R_EARTH),
        (60.0, 30.0, R_EARTH),
        (10.0, -40.0, R_EARTH),
        (40.0, 10.0, SEC_SHELL + 400e3),
        (60.0, 30.0, SEC_SHELL + 400e3),
    ]:
        div, _, B0 = numerical_div_curl(pysecs.T_df, SEC_LATLONR, lat, lon, r)
        # Compare against a characteristic gradient scale B / (100 km)
        assert abs(div) < 1e-7 * B0 / 1e5


def test_df_curl_free_off_shell():
    "curl B = 0 in the current-free regions for the divergence-free system."
    for lat, lon, r in [
        (40.0, 10.0, R_EARTH),
        (60.0, 30.0, R_EARTH),
        (40.0, 10.0, SEC_SHELL + 400e3),
    ]:
        _, curl, B0 = numerical_div_curl(pysecs.T_df, SEC_LATLONR, lat, lon, r)
        assert np.max(np.abs(curl)) < 1e-7 * B0 / 1e5


def test_cf_divergence_free_above():
    "div B = 0 for the curl-free system above the shell."
    for lat, lon, r in [
        (40.0, 10.0, SEC_SHELL + 400e3),
        (0.0, 60.0, SEC_SHELL + 800e3),
    ]:
        div, _, B0 = numerical_div_curl(pysecs.T_cf, SEC_LATLONR, lat, lon, r)
        assert abs(div) < 1e-7 * B0 / 1e5


def test_cf_curl_matches_distributed_fac():
    "curl B above the shell equals mu0 times the distributed FAC density."
    for lat, lon, r in [
        (40.0, 10.0, SEC_SHELL + 400e3),
        (0.0, 60.0, SEC_SHELL + 800e3),
    ]:
        _, curl, B0 = numerical_div_curl(pysecs.T_cf, SEC_LATLONR, lat, lon, r)
        # The distributed return FACs flow radially outward with density
        # I0 / (4 pi r^2), so (curl B)_r = mu0 / (4 pi r^2) per unit amplitude
        expected_curl_r = MU0 / (4 * np.pi * r**2)
        assert_allclose(curl[0], expected_curl_r, rtol=1e-4)
        # No theta/phi components of the curl (no toroidal currents)
        assert abs(curl[1]) < 1e-9 * B0 / 1e5
        assert abs(curl[2]) < 1e-9 * B0 / 1e5


def test_sheet_current_jump_conditions():
    "B(above) - B(below) across the shell equals mu0 (K x n) for df and cf."
    delta = 0.05  # meters on either side of the shell
    for Tfunc, Jfunc in [
        (pysecs.T_df, pysecs.J_df),
        (pysecs.T_cf, pysecs.J_cf),
    ]:
        for lat, lon in [(40.0, 10.0), (60.0, 100.0), (-30.0, -100.0)]:
            B_above = field_at(Tfunc, SEC_LATLONR, lat, lon, SEC_SHELL + delta)
            B_below = field_at(Tfunc, SEC_LATLONR, lat, lon, SEC_SHELL - delta)
            dB = B_above - B_below

            K = field_at(Jfunc, SEC_LATLONR, lat, lon, SEC_SHELL)
            # mu0 (K x n) with n = up = -z in (x north, y east, z down):
            # (Kx, Ky, Kz) x (0, 0, -1) = (-Ky, Kx, 0)
            expected = MU0 * np.array([-K[1], K[0], 0.0])
            assert_allclose(dB[:2], expected[:2], rtol=1e-6)
            # The radial component is continuous across the sheet
            assert abs(dB[2]) < 1e-6 * np.linalg.norm(expected)


def test_amm_viljanen_figure2_peak():
    """Br is exactly 10 nT below the pole for I0 = 10,000 A at 100 km altitude.

    Amm & Viljanen (1999), Fig. 2: "Br reaches a maximum (of exactly 10 nT
    for RI - r = 100 km) below the pole". The peak value is independent of
    the Earth radius used: Br(theta=0) = 1e-7 * I0 / (RI - r).
    """
    I0 = 10000.0
    sec = np.array([[0.0, 0.0, R_EARTH + 100e3]])
    obs = np.array([[0.0, 0.0, R_EARTH]])
    B = np.squeeze(pysecs.T_df(obs, sec)) * I0
    # Bz (down) is opposite Br (up); Btheta is zero directly below the pole
    assert_allclose(B[2], -10e-9, rtol=1e-9)
    assert B[0] == 0.0
    assert B[1] == 0.0


def test_amm_viljanen_figure2_btheta_minimum():
    """Btheta has a minimum at about 127 km ground distance from the pole.

    Amm & Viljanen (1999), Fig. 2 discussion.
    """
    I0 = 10000.0
    sec = np.array([[0.0, 0.0, R_EARTH + 100e3]])
    angles = np.linspace(0.01, 3.0, 4000)
    obs = np.zeros((len(angles), 3))
    obs[:, 1] = angles
    obs[:, 2] = R_EARTH
    B = np.squeeze(pysecs.T_df(obs, sec)) * I0
    # For observations due east of the pole, By is the theta component
    ground_dist = R_EARTH * np.deg2rad(angles)
    min_dist = ground_dist[np.argmin(B[:, 1])]
    assert 120e3 < min_dist < 135e3


def test_cf_line_current_limit():
    "Near the pole the CF field approaches that of an infinite line current."
    sec = np.array([[0.0, 0.0, SEC_SHELL]])
    r_obs = SEC_SHELL + 500e3
    for ang in [0.005, 0.01, 0.02]:
        obs = np.array([[0.0, ang, r_obs]])
        B = np.linalg.norm(np.squeeze(pysecs.T_cf(obs, sec)))
        rho = r_obs * np.deg2rad(ang)
        assert_allclose(B, MU0 / (2 * np.pi * rho), rtol=1e-6)


def test_round_trip_amplitude_recovery():
    "Fitting synthetic observations recovers the generating amplitudes."
    rng = np.random.default_rng(42)
    lats = np.linspace(35, 55, 6)
    lons = np.linspace(-10, 20, 8)
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
    sec_locs = np.column_stack(
        [lat_g.ravel(), lon_g.ravel(), np.full(lat_g.size, R_EARTH + 110e3)]
    )
    true_amps = rng.normal(0, 1e4, sec_locs.shape[0])

    nobs = 40
    obs_locs = np.column_stack(
        [rng.uniform(36, 54, nobs), rng.uniform(-9, 19, nobs), np.full(nobs, R_EARTH)]
    )
    obs_B = np.tensordot(true_amps, pysecs.T_df(obs_locs, sec_locs), (0, 2))

    secs = pysecs.SECS(sec_df_loc=sec_locs)
    secs.fit(obs_locs, obs_B[np.newaxis], epsilon=1e-10)
    assert_allclose(secs.sec_amps[0], true_amps, rtol=1e-6, atol=1e-6 * 1e4)

    # Predictions at held-out locations match the generating model
    npred = 25
    pred_locs = np.column_stack(
        [
            rng.uniform(38, 52, npred),
            rng.uniform(-5, 15, npred),
            np.full(npred, R_EARTH),
        ]
    )
    B_true = np.tensordot(true_amps, pysecs.T_df(pred_locs, sec_locs), (0, 2))
    assert_allclose(secs.predict_B(pred_locs), B_true, rtol=1e-8, atol=1e-8 * 1e-9)


def test_rotation_invariance_longitude():
    "Shifting every longitude by a constant leaves all transfer matrices unchanged."
    sec_locs = np.array(
        [[50.0, 10.0, SEC_SHELL], [45.0, 15.0, SEC_SHELL], [-20.0, 170.0, SEC_SHELL]]
    )
    obs_locs = np.array(
        [
            [48.0, 12.0, R_EARTH],
            [30.0, -60.0, R_EARTH],
            [52.0, 11.0, SEC_SHELL + 400e3],
        ]
    )
    for dlon in [37.5, 200.0]:
        sec_rot = sec_locs.copy()
        obs_rot = obs_locs.copy()
        # Keep longitudes in [-180, 180) to also exercise wrap-around
        sec_rot[:, 1] = (sec_rot[:, 1] + dlon + 180) % 360 - 180
        obs_rot[:, 1] = (obs_rot[:, 1] + dlon + 180) % 360 - 180
        for func in [pysecs.T_df, pysecs.T_cf, pysecs.J_df, pysecs.J_cf]:
            T0 = func(obs_locs, sec_locs)
            T1 = func(obs_rot, sec_rot)
            assert_allclose(T1, T0, rtol=1e-9, atol=1e-24)


def test_no_nans_at_theta_edge_cases():
    "The transfer functions stay finite at theta = 0 and theta = 180."
    sec = np.array([[0.0, 0.0, SEC_SHELL]])
    # Below the shell: pole and antipode
    obs_below = np.array([[0.0, 0.0, R_EARTH], [0.0, 180.0, R_EARTH]])
    for func in [pysecs.T_df, pysecs.T_cf]:
        T = func(obs_below, sec)
        assert np.all(np.isfinite(T))
    # Directly below the pole Btheta is zero (purely radial field)
    T = pysecs.T_df(obs_below, sec)
    assert T[0, 0, 0] == 0.0
    assert T[0, 1, 0] == 0.0
    # Above the shell at the antipode the CF field vanishes smoothly
    obs_above = np.array([[0.0, 180.0, SEC_SHELL + 400e3]])
    T = pysecs.T_cf(obs_above, sec)
    assert_allclose(T, 0.0, atol=1e-27)
