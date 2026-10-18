"""Tests for the KalmanSECS temporal state-space estimator."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import pysecs


R_EARTH = 6371e3


@pytest.fixture(scope="module")
def system():
    """A small df system with static truth and noisy observations."""
    rng = np.random.default_rng(11)
    lat_g, lon_g = np.meshgrid(
        np.linspace(-3, 3, 4), np.linspace(-3, 3, 4), indexing="ij"
    )
    sec_locs = np.column_stack(
        [lat_g.ravel(), lon_g.ravel(), np.full(16, R_EARTH + 110e3)]
    )
    true_amps = rng.normal(0, 1e4, 16)
    nobs = 12
    obs_locs = np.column_stack(
        [rng.uniform(-4, 4, nobs), rng.uniform(-4, 4, nobs), np.full(nobs, R_EARTH)]
    )
    T_obs = pysecs.T_df(obs_locs, sec_locs)
    B_true = np.tensordot(true_amps, T_obs, (0, 2))
    B_scale = np.max(np.abs(B_true))

    ntimes = 60
    times = np.arange(ntimes) * 60.0  # 1-minute cadence, in seconds
    noise_std = 0.05 * B_scale
    obs_B = B_true[np.newaxis] + rng.normal(0, noise_std, (ntimes, nobs, 3))
    obs_std = np.full_like(obs_B, noise_std)

    return {
        "sec_locs": sec_locs,
        "obs_locs": obs_locs,
        "T_obs": T_obs,
        "true_amps": true_amps,
        "B_scale": B_scale,
        "times": times,
        "noise_std": noise_std,
        "obs_B": obs_B,
        "obs_std": obs_std,
    }


def _kalman(sec_locs, **kwargs):
    return pysecs.KalmanSECS(pysecs.SECS(sec_df_loc=sec_locs), **kwargs)


def test_kalman_beats_snapshot_on_static_truth(system):
    """Temporal smoothing averages down noise that snapshot fits keep."""
    snap = pysecs.SECS(sec_df_loc=system["sec_locs"])
    snap.fit(
        system["obs_locs"], system["obs_B"], obs_std=system["obs_std"], epsilon=1e-6
    )
    err_snap = np.sqrt(np.mean((snap.sec_amps - system["true_amps"]) ** 2))

    kal = _kalman(system["sec_locs"], tau=3600.0, prior_std=3e4)
    kal.fit(system["obs_locs"], system["obs_B"], system["times"], system["obs_std"])
    err_kal = np.sqrt(np.mean((kal.sec_amps - system["true_amps"]) ** 2))

    assert err_kal < err_snap / 1.8


def test_smoother_improves_on_filter(system):
    """The RTS smoother uses future data, improving early estimates."""
    kal_f = _kalman(system["sec_locs"], tau=3600.0, prior_std=3e4, smoother=False)
    kal_f.fit(system["obs_locs"], system["obs_B"], system["times"], system["obs_std"])
    kal_s = _kalman(system["sec_locs"], tau=3600.0, prior_std=3e4)
    kal_s.fit(system["obs_locs"], system["obs_B"], system["times"], system["obs_std"])

    err_f = np.sqrt(np.mean((kal_f.sec_amps[:10] - system["true_amps"]) ** 2))
    err_s = np.sqrt(np.mean((kal_s.sec_amps[:10] - system["true_amps"]) ** 2))
    assert err_s < err_f
    # The final state has no future data, so filter and smoother agree there
    assert_allclose(kal_f.sec_amps[-1], kal_s.sec_amps[-1], rtol=1e-10)


def test_tau_zero_limit_is_per_step_ridge(system):
    """With tau -> 0 each step is an independent Bayesian ridge solution."""
    prior_std = 3e4
    kal = _kalman(system["sec_locs"], tau=1e-9, prior_std=prior_std)
    obs_B = system["obs_B"][:3]
    kal.fit(system["obs_locs"], obs_B, system["times"][:3], system["obs_std"][:3])

    H = system["T_obs"].reshape(-1, 16)
    prior_var = prior_std**2
    R = np.diag(np.full(H.shape[0], system["noise_std"] ** 2))
    for t in range(3):
        ridge = (
            prior_var * H.T @ np.linalg.solve(prior_var * H @ H.T + R, obs_B[t].ravel())
        )
        assert_allclose(kal.sec_amps[t], ridge, rtol=1e-8)


def test_gating_rejects_station_spike(system):
    """Innovation gating stops a station spike from steering the solution."""
    rng = np.random.default_rng(2)
    obs_B = system["obs_B"].copy()
    obs_B[30, 0, :] += 50 * system["B_scale"]

    pred_locs = np.column_stack(
        [rng.uniform(-3, 3, 20), rng.uniform(-3, 3, 20), np.full(20, R_EARTH)]
    )
    B_pred_true = np.tensordot(
        system["true_amps"], pysecs.T_df(pred_locs, system["sec_locs"]), (0, 2)
    )

    errs = {}
    for gate in [None, 4.0]:
        kal = _kalman(system["sec_locs"], tau=3600.0, prior_std=3e4, gate_sigma=gate)
        kal.fit(system["obs_locs"], obs_B, system["times"], system["obs_std"])
        pred = kal.predict_B(pred_locs)
        errs[gate] = np.max(np.abs(pred[30] - B_pred_true)) / system["B_scale"]

    # Ungated, the spike dominates the map at that time step
    assert errs[None] > 5.0
    # Gated, the spiked step stays near the noise level
    assert errs[4.0] < 0.3


def test_data_gap_bridged_with_growing_variance(system):
    """A full data gap is bridged smoothly and the variance grows in it."""
    obs_std = system["obs_std"].copy()
    obs_std[28:33] = np.inf

    kal = _kalman(system["sec_locs"], tau=600.0, prior_std=3e4)
    kal.fit(system["obs_locs"], system["obs_B"], system["times"], obs_std)

    assert np.all(np.isfinite(kal.sec_amps))
    # Uncertainty grows where there are no observations
    assert np.mean(kal.sec_amps_var[30]) > 5 * np.mean(kal.sec_amps_var[20])

    pred_locs = np.array([[0.5, 0.5, R_EARTH], [2.0, -1.0, R_EARTH]])
    pred, var = kal.predict_B(pred_locs, return_var=True)
    assert var.shape == pred.shape
    assert np.all(var > 0)
    assert np.mean(var[30]) > 5 * np.mean(var[20])


def test_datetime64_times_match_float_seconds(system):
    """datetime64 times give identical results to float seconds."""
    kal1 = _kalman(system["sec_locs"], tau=3600.0, prior_std=3e4)
    kal1.fit(system["obs_locs"], system["obs_B"], system["times"], system["obs_std"])

    dtimes = np.datetime64("2024-05-10T00:00") + (system["times"] * 1e6).astype(
        "timedelta64[us]"
    )
    kal2 = _kalman(system["sec_locs"], tau=3600.0, prior_std=3e4)
    kal2.fit(system["obs_locs"], system["obs_B"], dtimes, system["obs_std"])

    assert_allclose(kal1.sec_amps, kal2.sec_amps, rtol=1e-12)


def test_auto_prior_std(system):
    """The default prior scale is estimated from a snapshot fit."""
    kal = _kalman(system["sec_locs"], tau=3600.0)
    kal.fit(system["obs_locs"], system["obs_B"], system["times"], system["obs_std"])
    err = np.sqrt(np.mean((kal.sec_amps - system["true_amps"]) ** 2))

    snap = pysecs.SECS(sec_df_loc=system["sec_locs"])
    snap.fit(
        system["obs_locs"], system["obs_B"], obs_std=system["obs_std"], epsilon=1e-6
    )
    err_snap = np.sqrt(np.mean((snap.sec_amps - system["true_amps"]) ** 2))
    assert err < err_snap


def test_invalid_arguments(system):
    secs = pysecs.SECS(sec_df_loc=system["sec_locs"])
    with pytest.raises(ValueError, match="tau"):
        pysecs.KalmanSECS(secs, tau=0.0)
    with pytest.raises(ValueError, match="prior_std"):
        pysecs.KalmanSECS(secs, tau=60.0, prior_std=-1.0)
    with pytest.raises(ValueError, match="gate_sigma"):
        pysecs.KalmanSECS(secs, tau=60.0, gate_sigma=0.0)

    kal = _kalman(system["sec_locs"], tau=60.0, prior_std=3e4)
    with pytest.raises(ValueError, match="strictly increasing"):
        kal.fit(
            system["obs_locs"],
            system["obs_B"][:3],
            np.array([0.0, 60.0, 60.0]),
            system["obs_std"][:3],
        )
    with pytest.raises(ValueError, match="one entry per time step"):
        kal.fit(
            system["obs_locs"],
            system["obs_B"][:3],
            system["times"][:2],
            system["obs_std"][:3],
        )
    with pytest.raises(ValueError, match="return_var"):
        _kalman(system["sec_locs"], tau=60.0, prior_std=3e4).predict(
            np.array([[0.0, 0.0, R_EARTH]]), return_var=True
        )


def test_single_snapshot_2d_input(system):
    """A single 2-D snapshot works with a single time entry."""
    kal = _kalman(system["sec_locs"], tau=60.0, prior_std=3e4)
    kal.fit(system["obs_locs"], system["obs_B"][0], np.array([0.0]))
    assert kal.sec_amps.shape == (1, 16)
    assert np.all(np.isfinite(kal.sec_amps))


def test_missing_data_nan_ignored(system):
    """Non-finite observations are treated as missing, not propagated."""
    obs_B = system["obs_B"].copy()
    obs_B[10, 3, 1] = np.nan
    kal = _kalman(system["sec_locs"], tau=3600.0, prior_std=3e4)
    kal.fit(system["obs_locs"], obs_B, system["times"], system["obs_std"])
    assert np.all(np.isfinite(kal.sec_amps))
    assert np.all(np.isfinite(kal.sec_amps_var))


def test_fac_density(system):
    """FAC densities propagate the full amplitude covariance."""
    sec_locs = system["sec_locs"]
    kf = pysecs.KalmanSECS(
        pysecs.SECS(sec_df_loc=sec_locs, sec_cf_loc=sec_locs), tau=600.0, prior_std=1e4
    )
    with pytest.raises(ValueError, match="Call fit"):
        kf.fac_density(1e10)
    # Ground observations leave the cf amplitudes at their prior
    with pytest.warns(UserWarning, match="Fukushima") as record:
        kf.fit(system["obs_locs"], system["obs_B"], system["times"], system["obs_std"])
    assert record[0].filename == __file__

    area = np.full(len(sec_locs), 1e10)
    fac, var = kf.fac_density(area, return_var=True)
    D = kf.secs._fac_operator(area)
    assert_allclose(fac, kf.sec_amps @ D.T)
    expected_var = np.einsum("ij,tjk,ik->ti", D, kf.sec_amps_cov, D)
    assert_allclose(var, expected_var)


def _current_system(system):
    """df + cf truth on the system grid with ground and shell observations."""
    rng = np.random.default_rng(5)
    sec_locs = system["sec_locs"]
    truth = pysecs.SECS(sec_df_loc=sec_locs, sec_cf_loc=sec_locs)
    truth.sec_amps = rng.normal(0, 1e4, (1, truth.nsec))
    J_locs = system["obs_locs"].copy()
    J_locs[:, 2] = sec_locs[0, 2]
    # Denser current observations so that they determine all amplitudes
    shift = np.array([0.5, 0.5, 0.0])
    J_locs = np.vstack([J_locs, J_locs + shift, J_locs - shift])
    return truth, J_locs


def test_fit_J_matches_snapshot_without_noise(system):
    """With uninformative dynamics, fit_J recovers the noise-free truth."""
    truth, J_locs = _current_system(system)
    obs_J = np.repeat(truth.predict_J(J_locs)[np.newaxis, :, :2], 3, axis=0)
    J_std = np.full_like(obs_J, 1e-6)

    kf = pysecs.KalmanSECS(
        pysecs.SECS(sec_df_loc=truth.sec_df_loc, sec_cf_loc=truth.sec_cf_loc),
        tau=60.0,
        prior_std=1e6,
    )
    assert kf.fit_J(J_locs, obs_J, np.arange(3) * 60.0, J_std) is kf
    assert_allclose(kf.sec_amps, np.repeat(truth.sec_amps, 3, axis=0), rtol=1e-3)


def test_joint_fit_constrains_curl_free(system):
    """Adding currents to ground data constrains the cf amplitudes."""
    truth, J_locs = _current_system(system)
    ntimes = 5
    times = np.arange(ntimes) * 60.0
    obs_B = np.repeat(truth.predict_B(system["obs_locs"])[np.newaxis], ntimes, axis=0)
    obs_J = np.repeat(truth.predict_J(J_locs)[np.newaxis, :, :2], ntimes, axis=0)
    B_std = np.full_like(obs_B, 1e-12)
    J_std = np.full_like(obs_J, 1e-6)
    # A missing current observation is skipped at that time step
    obs_J[2, 0] = np.nan

    def kalman():
        return pysecs.KalmanSECS(
            pysecs.SECS(sec_df_loc=truth.sec_df_loc, sec_cf_loc=truth.sec_cf_loc),
            tau=600.0,
            prior_std=1e5,
        )

    with pytest.warns(UserWarning, match="Fukushima"):
        ground = kalman().fit(system["obs_locs"], obs_B, times, B_std)
    joint = kalman().fit(
        system["obs_locs"], obs_B, times, B_std, J_loc=J_locs, obs_J=obs_J, J_std=J_std
    )

    ncf = len(truth.sec_cf_loc)
    true_cf = truth.sec_amps[0, -ncf:]
    assert_allclose(joint.sec_amps[:, -ncf:], np.tile(true_cf, (ntimes, 1)), rtol=1e-3)
    # From the ground alone the cf amplitudes stay at their prior mean
    assert_allclose(ground.sec_amps[:, -ncf:], 0.0, atol=1e-6)
    # ... with their prior uncertainty, which the currents reduce
    assert np.all(joint.sec_amps_var[:, -ncf:] < ground.sec_amps_var[:, -ncf:])

    # The joint estimate reproduces the observed currents and FACs
    pred_J = joint.predict_J(J_locs)[:, :, :2]
    assert_allclose(pred_J[1], obs_J[1], atol=1e-3 * np.max(np.abs(obs_J[1])))
    area = 1e10
    assert_allclose(joint.fac_density(area)[0], truth.fac_density(area), rtol=1e-3)


def test_joint_fit_bad_arguments(system):
    """Joint Kalman fits validate their inputs like SECS.fit."""
    kf = _kalman(system["sec_locs"], tau=60.0)
    args = (system["obs_locs"], system["obs_B"], system["times"], system["obs_std"])
    with pytest.raises(ValueError, match="must be given together"):
        kf.fit(*args, obs_J=system["obs_B"])
    with pytest.raises(ValueError, match="requires both obs_std and J_std"):
        kf.fit(*args, J_loc=system["obs_locs"], obs_J=system["obs_B"])


def test_fit_J_auto_prior_std(system):
    """prior_std is estimated from a snapshot fit of the currents."""
    truth, J_locs = _current_system(system)
    obs_J = np.repeat(truth.predict_J(J_locs)[np.newaxis, :, :2], 3, axis=0)
    obs_J[1, 0] = np.nan

    kf = pysecs.KalmanSECS(
        pysecs.SECS(sec_df_loc=truth.sec_df_loc, sec_cf_loc=truth.sec_cf_loc),
        tau=60.0,
    )
    with pytest.warns(UserWarning, match="not located on a SEC shell") as record:
        kf.fit_J(J_locs + np.array([0.0, 0.0, 1.0]), obs_J, np.arange(3) * 60.0)
    assert record[0].filename == __file__
    kf.fit_J(J_locs, obs_J, np.arange(3) * 60.0)
    assert np.all(np.isfinite(kf.sec_amps))
    assert np.all(kf.sec_amps_var > 0)
