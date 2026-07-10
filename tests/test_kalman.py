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
