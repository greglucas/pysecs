import numpy as np
import pytest
from numpy.testing import assert_allclose

from pysecs import SECS, calc_angular_distance, make_grid, make_image_grid


R_EARTH = 6378e3
R_IONOSPHERE = R_EARTH + 110e3


def _stations():
    # A small, irregular network so nearest-neighbor spacing isn't trivial
    lat = np.array([60.0, 58.0, 62.0, 59.0, 61.5])
    lon = np.array([5.0, 10.0, 12.0, -2.0, 6.0])
    return np.column_stack([lat, lon, np.full(5, R_EARTH)])


def test_grid_covers_and_pads_stations():
    "The grid should extend beyond the station bounding box."
    obs = _stations()
    grid = make_grid(obs, R_IONOSPHERE)

    assert grid.shape[-1] == 3
    assert_allclose(grid[:, 2], R_IONOSPHERE)
    assert grid[:, 0].min() < obs[:, 0].min()
    assert grid[:, 0].max() > obs[:, 0].max()
    assert grid[:, 1].min() < obs[:, 1].min()
    assert grid[:, 1].max() > obs[:, 1].max()


def test_default_spacing_from_station_separation():
    "Default spacing should scale with the median nearest-neighbor distance."
    obs = _stations()
    coarse = make_grid(obs, R_IONOSPHERE, spacing=4.0, padding=0.0, min_distance=0)
    fine = make_grid(obs, R_IONOSPHERE, spacing=0.5, padding=0.0, min_distance=0)
    default = make_grid(obs, R_IONOSPHERE, padding=0.0, min_distance=0)

    assert len(fine) > len(default) > len(coarse)


def test_explicit_spacing_tuple_sets_lat_lon_independently():
    "A (lat, lon) tuple should set each spacing directly, with no cos-scaling."
    obs = _stations()
    grid = make_grid(obs, R_IONOSPHERE, spacing=(2.0, 5.0), padding=0.0, min_distance=0)
    lats = np.unique(grid[:, 0])
    lons = np.unique(grid[:, 1])
    assert_allclose(np.diff(lats), 2.0)
    assert_allclose(np.diff(lons), 5.0)


def test_padding_expands_bbox_by_requested_amount():
    obs = _stations()
    no_pad = make_grid(obs, R_IONOSPHERE, spacing=1.0, padding=0.0, min_distance=0)
    padded = make_grid(obs, R_IONOSPHERE, spacing=1.0, padding=5.0, min_distance=0)

    assert_allclose(no_pad[:, 0].min(), obs[:, 0].min())
    assert_allclose(no_pad[:, 0].max(), obs[:, 0].max())
    assert padded[:, 0].min() < no_pad[:, 0].min()
    assert padded[:, 0].max() > no_pad[:, 0].max()


def _coincident_stations():
    # With padding=0 and spacing=1.0 the grid is anchored exactly at
    # (60, 5), guaranteeing a coincident node for the nudge tests below.
    return np.array([[60.0, 5.0, R_EARTH], [61.0, 6.0, R_EARTH]])


def test_min_distance_nudges_coincident_node():
    "A grid node placed exactly on a station should be pushed away."
    obs = _coincident_stations()
    grid_off = make_grid(obs, R_IONOSPHERE, spacing=1.0, padding=0.0, min_distance=0)
    dist_off = np.rad2deg(calc_angular_distance(grid_off[:, :2], obs[:, :2]))
    assert dist_off.min() < 1e-9  # confirms the coincidence actually occurs

    grid = make_grid(obs, R_IONOSPHERE, spacing=1.0, padding=0.0, min_distance=0.1)
    dist = np.rad2deg(calc_angular_distance(grid[:, :2], obs[:, :2]))
    assert dist.min() >= 0.1 - 1e-8


def test_min_distance_zero_disables_nudging():
    obs = _coincident_stations()
    grid_off = make_grid(obs, R_IONOSPHERE, spacing=1.0, padding=0.0, min_distance=0)
    dist_deg = np.rad2deg(calc_angular_distance(grid_off[:, :2], obs[:, :2]))
    assert dist_deg.min() < 1e-9


def test_requires_explicit_spacing_with_one_station():
    obs = np.array([[60.0, 5.0, R_EARTH]])
    with pytest.raises(ValueError, match="spacing must be given explicitly"):
        make_grid(obs, R_IONOSPHERE)


def test_requires_three_columns():
    with pytest.raises(ValueError, match="3 columns"):
        make_grid(np.array([[60.0, 5.0]]), R_IONOSPHERE, spacing=1.0)


def test_nonpositive_spacing_raises():
    obs = _stations()
    with pytest.raises(ValueError, match="positive"):
        make_grid(obs, R_IONOSPHERE, spacing=0.0)


def test_make_image_grid_changes_only_radius():
    obs = _stations()
    grid = make_grid(obs, R_IONOSPHERE, spacing=2.0)
    image = make_image_grid(grid, R_EARTH - 500e3)

    assert_allclose(image[:, :2], grid[:, :2])
    assert_allclose(image[:, 2], R_EARTH - 500e3)
    # Independent copy: mutating the image must not affect the original grid
    image[0, 0] = -999.0
    assert grid[0, 0] != -999.0


def test_from_observations_default_is_df_only():
    obs = _stations()
    secs = SECS.from_observations(obs, R_IONOSPHERE, spacing=2.0)
    assert secs.has_df
    assert not secs.has_cf


def test_from_observations_cf_only():
    obs = _stations()
    secs = SECS.from_observations(obs, R_IONOSPHERE, spacing=2.0, df=False, cf=True)
    assert not secs.has_df
    assert secs.has_cf


def test_from_observations_both():
    obs = _stations()
    secs = SECS.from_observations(obs, R_IONOSPHERE, spacing=2.0, df=True, cf=True)
    assert secs.has_df
    assert secs.has_cf
    assert secs.nsec == 2 * len(secs.sec_df_loc)


def test_from_observations_requires_df_or_cf():
    obs = _stations()
    with pytest.raises(ValueError, match="df or cf"):
        SECS.from_observations(obs, R_IONOSPHERE, spacing=2.0, df=False, cf=False)


def test_from_observations_recovers_smooth_field():
    "End-to-end sanity check: fit and predict back at the stations."
    rng = np.random.default_rng(0)
    obs = _stations()

    truth_grid = make_grid(obs, R_IONOSPHERE, spacing=1.0)
    truth = SECS(sec_df_loc=truth_grid)
    truth.sec_amps = rng.normal(0, 1e4, (1, len(truth_grid)))
    B_obs = truth.predict(obs)

    secs = SECS.from_observations(obs, R_IONOSPHERE)
    secs.fit(obs, B_obs)
    B_fit = secs.predict(obs)

    assert_allclose(B_fit, np.squeeze(B_obs), atol=1e-6, rtol=1e-3)
