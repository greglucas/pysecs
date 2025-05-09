import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import pysecs
from pysecs.secs import _calc_angular_distance_and_bearing


R_EARTH = 6378e3


def test_angular_distance():
    "Test the angular distance formula."
    latlon1 = np.array([[0.0, 0.0]])
    latlon2 = np.array([[0.0, 0.0], [0.0, 90], [-90.0, 0.0], [0.0, 180.0]])
    assert_array_equal(
        pysecs.calc_angular_distance(latlon1, latlon2),
        np.deg2rad([[0.0, 90.0, 90.0, 180.0]]),
    )


def test_bearing():
    "Test the cardinal directions."
    latlon1 = np.array([[0.0, 0.0]])
    latlon2 = np.array(
        [[0.0, 90.0], [90.0, 0.0], [90.0, 45.0], [0.0, -90.0], [-90, 0.0]]
    )
    assert_array_equal(
        pysecs.calc_bearing(latlon1, latlon2),
        np.deg2rad([[0.0, 90.0, 90.0, 180.0, -90]]),
    )


def test_distance_and_bearing():
    "Test the combined function."
    latlon1 = np.array([[0.0, 0.0]])
    latlon2 = np.array(
        [[0.0, 90.0], [90.0, 0.0], [90.0, 45.0], [0.0, -90.0], [-90, 0.0]]
    )
    theta, alpha = _calc_angular_distance_and_bearing(latlon1, latlon2)
    assert_array_equal(theta, pysecs.calc_angular_distance(latlon1, latlon2))
    assert_array_equal(alpha, pysecs.calc_bearing(latlon1, latlon2))


def test_divergence_free_magnetic_directions():
    "Make sure the divergence free magnetic field angles are correct"
    # Place the SEC at the equator
    sec_r = R_EARTH + 100
    sec_latlonr = np.array([[0.0, 0.0, sec_r]])
    # Going around in a circle from the point
    obs_latlonr = np.array(
        [
            [5.0, 0.0, R_EARTH],
            [0.0, 5.0, R_EARTH],
            [-5, 0.0, R_EARTH],
            [0.0, -5.0, R_EARTH],
        ]
    )

    B = np.squeeze(pysecs.T_df(obs_latlonr, sec_latlonr))

    angles = np.arctan2(B[:, 0], B[:, 1])
    # southward, westward, northward, eastward
    expected_angles = np.deg2rad([-90, 180.0, 90.0, 0.0])
    assert_allclose(angles, expected_angles, rtol=1e-10, atol=1e-10)


def test_divergence_free_magnetic_magnitudes_obs_under():
    "Make sure the divergence free magnetic amplitudes are correct."
    # Place the SEC at the North Pole
    sec_r = R_EARTH + 100
    sec_latlonr = np.array([[0.0, 0.0, sec_r]])
    # Going out in an angle from the SEC (in longitude)
    angles = np.linspace(0.1, 180)
    obs_r = R_EARTH
    obs_latlonr = np.zeros((*angles.shape, 3))
    obs_latlonr[:, 1] = angles
    obs_latlonr[:, 2] = obs_r

    B = np.squeeze(pysecs.T_df(obs_latlonr, sec_latlonr))

    # All x components should be zero (angles goes around the equator and all
    # quantities should be parallel to that)
    assert_allclose(np.zeros(angles.shape), B[:, 0], atol=1e-16)

    # Actual magnitude
    mu0 = 4 * np.pi * 1e-7

    # simplify calculations by storing this ratio
    x = obs_r / sec_r

    sin_theta = np.sin(np.deg2rad(angles))
    cos_theta = np.cos(np.deg2rad(angles))
    factor = 1.0 / np.sqrt(1 - 2 * x * cos_theta + x**2)

    # Amm & Viljanen: Equation 9
    Br = mu0 / (4 * np.pi * obs_r) * (factor - 1)
    # Bz in opposite direction of Br
    assert_allclose(-Br, B[:, 2])

    # Amm & Viljanen: Equation 10
    Btheta = -mu0 / (4 * np.pi * obs_r) * (factor * (x - cos_theta) + cos_theta)
    Btheta = np.divide(
        Btheta, sin_theta, out=np.zeros_like(sin_theta), where=sin_theta != 0
    )
    assert_allclose(Btheta, B[:, 1])


def test_divergence_free_magnetic_magnitudes_obs_over():
    "Make sure the divergence free magnetic amplitudes are correct."
    # Place the SEC at the North Pole
    sec_r = R_EARTH
    sec_latlonr = np.array([[0.0, 0.0, sec_r]])
    # Going out in an angle from the SEC (in longitude)
    angles = np.linspace(0.1, 180)
    obs_r = R_EARTH + 100
    obs_latlonr = np.zeros((*angles.shape, 3))
    obs_latlonr[:, 1] = angles
    obs_latlonr[:, 2] = obs_r

    B = np.squeeze(pysecs.T_df(obs_latlonr, sec_latlonr))

    # All x components should be zero (angles goes around the equator and all
    # quantities should be parallel to that)
    assert_allclose(np.zeros(angles.shape), B[:, 0], atol=1e-16)

    # Actual magnitude
    mu0 = 4 * np.pi * 1e-7
    x = sec_r / obs_r

    sin_theta = np.sin(np.deg2rad(angles))
    cos_theta = np.cos(np.deg2rad(angles))

    # Amm & Viljanen: Equation A.7
    Br = (
        mu0
        * x
        / (4 * np.pi * obs_r)
        * (1.0 / np.sqrt(1 - 2 * x * cos_theta + x**2) - 1)
    )
    # Bz in opposite direction of Br
    assert_allclose(-Br, B[:, 2])

    # Amm & Viljanen: Equation A.8
    Btheta = (
        -mu0
        / (4 * np.pi * obs_r)
        * (
            (obs_r - sec_r * cos_theta)
            / np.sqrt(obs_r**2 - 2 * obs_r * sec_r * cos_theta + sec_r**2)
            - 1
        )
    )
    Btheta = np.divide(
        Btheta, sin_theta, out=np.zeros_like(sin_theta), where=sin_theta != 0
    )
    assert_allclose(Btheta, B[:, 1])


def test_outside_current_plane():
    "Make sure all currents outside the SEC plane are 0."
    sec_r = R_EARTH + 100
    sec_latlonr = np.array([[0.0, 0.0, sec_r]])
    # Above and below the plane, also on and off the SEC point
    obs_latlonr = np.array(
        [
            [0.0, 0.0, sec_r - 100.0],
            [0.0, 0.0, sec_r + 100.0],
            [5, 0.0, sec_r - 100.0],
            [5.0, 0.0, sec_r + 100.0],
        ]
    )

    # df currents
    J = np.squeeze(pysecs.J_df(obs_latlonr, sec_latlonr))
    assert np.all(J == 0.0)
    # cf currents
    J = np.squeeze(pysecs.J_cf(obs_latlonr, sec_latlonr))
    assert np.all(J == 0.0)


def test_divergence_free_current_directions():
    "Make sure the divergence free current angles are correct."
    # Place the SEC at the equator
    sec_r = R_EARTH + 100
    sec_latlonr = np.array([[0.0, 0.0, sec_r]])
    # Going around in a circle from the point
    obs_latlonr = np.array(
        [[5.0, 0.0, sec_r], [0.0, 5.0, sec_r], [-5, 0.0, sec_r], [0.0, -5.0, sec_r]]
    )

    J = np.squeeze(pysecs.J_df(obs_latlonr, sec_latlonr))

    angles = np.arctan2(J[:, 0], J[:, 1])
    # westward, northward, eastward, southward
    expected_angles = np.deg2rad([-180.0, 90.0, 0.0, -90.0])
    assert_allclose(angles, expected_angles, atol=1e-16)


def test_divergence_free_current_magnitudes():
    "Make sure the divergence free current amplitudes are correct."
    # Place the SEC at the North Pole
    sec_r = R_EARTH + 100
    sec_latlonr = np.array([[0.0, 0.0, sec_r]])
    # Going out in an angle from the SEC (in longitude)
    angles = np.linspace(0.1, 180)
    obs_latlonr = np.zeros((*angles.shape, 3))
    obs_latlonr[:, 1] = angles
    obs_latlonr[:, 2] = sec_r

    J = np.squeeze(pysecs.J_df(obs_latlonr, sec_latlonr))

    # Make sure all radial components are zero in this system
    assert np.all(J[:, 2] == 0.0)

    # Also all y components (angles goes around the equator and all
    # quantities should be perpendicular to that)
    assert_allclose(np.zeros(angles.shape), J[:, 1], atol=1e-16)

    # Actual magnitude
    tan_theta2 = np.tan(np.deg2rad(angles / 2))
    J_test = 1.0 / (4 * np.pi * sec_r)
    J_test = np.divide(
        J_test,
        tan_theta2,
        out=np.ones_like(tan_theta2) * np.inf,
        where=tan_theta2 != 0.0,
    )

    assert_allclose(J_test, J[:, 0], atol=1e-16)


def test_curl_free_current_directions():
    "Make sure the curl free current angles are correct."
    # Place the SEC at the equator
    sec_r = R_EARTH + 100
    sec_latlonr = np.array([[0.0, 0.0, sec_r]])
    # Going around in a circle from the point
    obs_latlonr = np.array(
        [[5.0, 0.0, sec_r], [0.0, 5.0, sec_r], [-5, 0.0, sec_r], [0.0, -5.0, sec_r]]
    )

    J = np.squeeze(pysecs.J_cf(obs_latlonr, sec_latlonr))

    angles = np.arctan2(J[:, 0], J[:, 1])
    # pointing out from the SEC direction to OBS direction.
    # northward, eastward, southward, westward
    expected_angles = np.deg2rad([90.0, 0.0, -90.0, -180])
    assert_allclose(angles, expected_angles, atol=1e-15)


def test_curl_free_current_magnitudes():
    "Make sure the curl free current amplitudes are correct."
    # Place the SEC at the North Pole
    sec_r = R_EARTH + 100
    sec_latlonr = np.array([[0.0, 0.0, sec_r]])
    # Going out in an angle from the SEC (in longitude)
    angles = np.linspace(0.1, 180)
    obs_latlonr = np.zeros((*angles.shape, 3))
    obs_latlonr[:, 1] = angles
    obs_latlonr[:, 2] = sec_r

    J = np.squeeze(pysecs.J_cf(obs_latlonr, sec_latlonr))

    # Make sure all radial components are oppositely directed
    radial_component = 1.0 / (4 * np.pi * sec_r**2)
    assert np.all(J[:, 2] == radial_component)

    # All x components should be zero (angles goes around the equator and all
    # quantities should be parallel to that)
    # (ambiguous 0 degree angle so ignore the first input)
    assert_allclose(np.zeros(angles.shape), J[:, 0], atol=1e-16)

    # Actual magnitude
    tan_theta2 = np.tan(np.deg2rad(angles / 2))
    J_test = 1.0 / (4 * np.pi * sec_r)
    J_test = np.divide(
        J_test,
        tan_theta2,
        out=np.ones_like(tan_theta2) * np.inf,
        where=tan_theta2 != 0.0,
    )

    assert_allclose(J_test, J[:, 1], atol=1e-16)


def test_curl_free_magnetic_magnitudes():
    "Make sure the curl free magnetic amplitudes are correct."
    # TODO: Update this test once T_cf is implemented
    # Place the SEC at the North Pole
    sec_r = R_EARTH + 100
    sec_latlonr = np.array([[0.0, 0.0, sec_r]])
    # Going out in an angle from the SEC (in longitude)
    angles = np.linspace(0.1, 180)
    obs_latlonr = np.zeros((*angles.shape, 3))
    obs_latlonr[:, 1] = angles
    obs_latlonr[:, 2] = sec_r

    with pytest.raises(NotImplementedError, match="Curl Free Magnetic"):
        pysecs.T_cf(obs_latlonr, sec_latlonr)


def test_empty_object():
    "Testing empty secs object creation failure."
    with pytest.raises(ValueError, match="Must specify"):
        pysecs.SECS()


def test_list_numpy():
    "Make sure creation with numpy and list produce the same locations."
    x2d = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    x2d_np = np.array(x2d)
    secs_list2 = pysecs.SECS(sec_df_loc=x2d, sec_cf_loc=x2d)
    secs_np2 = pysecs.SECS(sec_df_loc=x2d_np, sec_cf_loc=x2d_np)
    assert secs_list2.nsec == 4  # 2 df + 2 cf
    assert secs_list2.nsec == secs_np2.nsec
    assert_array_equal(secs_list2.sec_df_loc, secs_np2.sec_df_loc)
    assert_array_equal(secs_list2.sec_cf_loc, secs_np2.sec_cf_loc)


def test_sec_bad_shape():
    """Test bad input shape."""
    # Wrong dimensions
    x = np.array([[1, 0], [1, 0]])
    with pytest.raises(ValueError, match="SEC DF locations"):
        pysecs.SECS(sec_df_loc=x)
    with pytest.raises(ValueError, match="SEC CF locations"):
        pysecs.SECS(sec_cf_loc=x)


def test_one_sec():
    """Test 1-dimensional input location gets mapped properly."""
    x1d = np.array([1, 0, 0])
    secs = pysecs.SECS(sec_df_loc=x1d)
    assert secs.nsec == 1
    secs = pysecs.SECS(sec_cf_loc=x1d)
    assert secs.nsec == 1


def test_fit_unit_currents():
    """Test the unit current function."""
    # divergence free
    secs = pysecs.SECS(sec_df_loc=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    secs.fit_unit_currents()
    assert_array_equal(np.ones((1, 2)), secs.sec_amps)

    # curl free
    secs = pysecs.SECS(sec_cf_loc=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    secs.fit_unit_currents()
    assert_array_equal(np.ones((1, 2)), secs.sec_amps)

    # divergence free + curl free
    secs = pysecs.SECS(
        sec_df_loc=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        sec_cf_loc=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    secs.fit_unit_currents()
    assert_array_equal(np.ones((1, 4)), secs.sec_amps)


def test_fit_bad_obsloc():
    """Test bad observation locations input."""
    secs = pysecs.SECS(sec_df_loc=[1.0, 0.0, R_EARTH + 1e6])
    obs_loc = np.array([[0, 0]])
    obs_B = np.array([[1, 1, 1]])
    with pytest.raises(ValueError, match="Observation locations"):
        secs.fit(obs_loc, obs_B)


def test_fit_one_time():
    """One timestep fit."""
    secs = pysecs.SECS(
        sec_df_loc=[[1.0, 0.0, R_EARTH + 1e6], [-1.0, 0.0, R_EARTH + 1e6]]
    )
    obs_loc = np.array([[0, 0, R_EARTH]])
    obs_B = np.array([[1, 1, 1]])
    secs.fit(obs_loc, obs_B, epsilon=0)
    assert_allclose([[6.40594202e13, -7.41421248e13]], secs.sec_amps)


def test_fit_multi_time():
    """Multi timestep fits."""
    secs = pysecs.SECS(
        sec_df_loc=[[1.0, 0.0, R_EARTH + 1e6], [-1.0, 0.0, R_EARTH + 1e6]]
    )
    obs_loc = np.array([[0, 0, R_EARTH]])
    obs_B = np.ones((2, 1, 3))
    obs_B[1, :, :] *= 2
    secs.fit(obs_loc, obs_B, epsilon=0)
    arr = np.array([6.40594202e13, -7.41421248e13])
    expected = np.array([arr, 2 * arr])
    assert_allclose(expected, secs.sec_amps)


def test_fit_obs_std():
    """Test that variance on observations changes the results."""
    secs = pysecs.SECS(
        sec_df_loc=[[1.0, 0.0, R_EARTH + 1e6], [-1.0, 0.0, R_EARTH + 1e6]]
    )
    obs_loc = np.array([[0, 0, R_EARTH]])
    obs_B = np.ones((2, 1, 3))
    obs_B[1, :, :] *= 2
    obs_std = np.ones(obs_B.shape)
    # Remove the z component from the fit of the second timestep
    obs_std[1, :, 2] = np.inf
    secs.fit(obs_loc, obs_B, obs_std=obs_std)
    expected = np.array([[6.40594202e13, -7.41421248e13], [1.382015e14, -1.382015e14]])
    assert_allclose(expected, secs.sec_amps, rtol=1e-6)


@pytest.mark.parametrize("mode", ["relative", "variance"])
def test_fit_epsilon(mode):
    """Test that epsilon removes some components."""
    secs = pysecs.SECS(
        sec_df_loc=[[1.0, 0.0, R_EARTH + 1e6], [-1.0, 0.0, R_EARTH + 1e6]]
    )
    obs_loc = np.array([[0, 0, R_EARTH]])
    obs_B = np.ones((2, 1, 3))
    obs_B[1, :, :] *= 2
    obs_std = np.ones(obs_B.shape)
    # Remove the z component from the fit of the second timestep
    obs_std[1, :, 2] = np.inf
    secs.fit(obs_loc, obs_B, obs_std=obs_std, epsilon=0.8, mode=mode)
    expected = np.array([[-5.041352e12, -5.041352e12], [1.382015e14, -1.382015e14]])
    assert_allclose(expected, secs.sec_amps, rtol=1e-6)


def test_bad_mode():
    """Test bad input to fit."""
    secs = pysecs.SECS(sec_df_loc=[[1.0, 0.0, R_EARTH + 1e6]])
    obs_loc = np.array([[0, 0, R_EARTH]])
    obs_B = np.ones((2, 1, 3))
    obs_B[1, :, :] *= 2
    with pytest.raises(ValueError, match="Unknown SVD filtering mode"):
        secs.fit(obs_loc, obs_B, epsilon=0.8, mode="bad_mode")


def test_bad_predict():
    """Test bad input to predict."""
    secs = pysecs.SECS(
        sec_df_loc=[[1.0, 0.0, R_EARTH + 1e6], [-1.0, 0.0, R_EARTH + 1e6]]
    )

    # Calling predict with the wrong shape
    pred_loc = np.array([[0, 0]])
    with pytest.raises(ValueError, match="Prediction locations"):
        secs.predict(pred_loc)

    # Calling predict before fitting
    pred_loc = np.array([[0, 0, R_EARTH]])
    with pytest.raises(ValueError, match="There are no currents associated"):
        secs.predict(pred_loc)


def test_predictB():
    """Test that epsilon removes some components."""
    secs = pysecs.SECS(
        sec_df_loc=[
            [1, 0, R_EARTH + 1e6],
            [-1, 0, R_EARTH + 1e6],
            [-1, 1, R_EARTH + 1e6],
            [1, 1, R_EARTH + 1e6],
            [0, 1, R_EARTH + 1e6],
            [0, -1, R_EARTH + 1e6],
            [-1, -1, R_EARTH + 1e6],
            [1, -1, R_EARTH + 1e6],
        ]
    )
    obs_loc = np.array([[0, 0, R_EARTH]])
    obs_B = np.ones((2, 1, 3))
    obs_B[1, :, :] *= 2
    secs.fit(obs_loc, obs_B, epsilon=0)

    # Predict at the same observatory location
    B_pred = secs.predict(obs_loc)
    assert_allclose(obs_B[:, 0, :], B_pred)

    # Call the predict_B method directly
    assert_allclose(secs.predict_B(obs_loc), secs.predict(obs_loc))


def test_predictJ():
    """Test the current sheet predictions (df)."""
    secs = pysecs.SECS(
        sec_df_loc=[
            [1, 0, R_EARTH + 1e6],
            [-1, 0, R_EARTH + 1e6],
            [-1, 1, R_EARTH + 1e6],
            [1, 1, R_EARTH + 1e6],
            [0, 1, R_EARTH + 1e6],
            [0, -1, R_EARTH + 1e6],
            [-1, -1, R_EARTH + 1e6],
            [1, -1, R_EARTH + 1e6],
        ]
    )
    obs_loc = np.array([[0, 0, R_EARTH]])
    obs_B = np.ones((2, 1, 3))
    obs_B[1, :, :] *= 2
    secs.fit(obs_loc, obs_B, epsilon=0)

    # Currents only on the SECS surface
    J_pred = secs.predict(obs_loc, J=True)
    assert_allclose(np.zeros((2, 3)), J_pred)

    # Move up to the current sheet
    pred_loc = np.array([[0, 0, R_EARTH + 1e6]])
    J_pred = secs.predict(pred_loc, J=True)
    arr = np.array([-1.148475e08, 1.148417e08, 0.000000e00])
    expected = np.array([arr, 2 * arr])
    assert_allclose(expected, J_pred, rtol=1e-6)

    # Use the predict_J function call directly
    assert_allclose(secs.predict_J(pred_loc), secs.predict(pred_loc, J=True))


def test_predictJ_cf():
    """Test the current sheet predictions (cf)."""
    sec_loc = np.array(
        [
            [1, 0, R_EARTH + 1e6],
            [-1, 0, R_EARTH + 1e6],
            [-1, 1, R_EARTH + 1e6],
            [1, 1, R_EARTH + 1e6],
            [0, 1, R_EARTH + 1e6],
            [0, -1, R_EARTH + 1e6],
            [-1, -1, R_EARTH + 1e6],
            [1, -1, R_EARTH + 1e6],
        ]
    )
    secs = pysecs.SECS(sec_cf_loc=sec_loc)
    obs_loc = np.array([[0, 0, R_EARTH]])
    secs.fit_unit_currents()

    # Currents only on the SECS surface
    J_pred = secs.predict(obs_loc, J=True)
    assert_allclose(np.zeros(3), J_pred)

    # Move up to the current sheet
    pred_loc = np.array([[0, 0, R_EARTH + 1e6]])
    J_pred = secs.predict(pred_loc, J=True)
    expected = np.array([0, 0, 1.169507e-14])
    assert_allclose(expected, J_pred, rtol=1e-6, atol=1e-10)

    # Use the predict_J function call directly
    assert_allclose(secs.predict_J(pred_loc), secs.predict(pred_loc, J=True))


def test_predictJ_cf_df():
    """Test the current sheet predictions (cf+df)."""
    sec_loc = np.array(
        [
            [1, 0, R_EARTH + 1e6],
            [-1, 0, R_EARTH + 1e6],
            [-1, 1, R_EARTH + 1e6],
            [1, 1, R_EARTH + 1e6],
            [0, 1, R_EARTH + 1e6],
            [0, -1, R_EARTH + 1e6],
            [-1, -1, R_EARTH + 1e6],
            [1, -1, R_EARTH + 1e6],
        ]
    )
    secs = pysecs.SECS(sec_df_loc=sec_loc, sec_cf_loc=sec_loc)
    obs_loc = np.array([[0, 0, R_EARTH]])
    secs.fit_unit_currents()

    # Currents only on the SECS surface
    J_pred = secs.predict(obs_loc, J=True)
    assert_allclose(np.zeros(3), J_pred)

    # Move up to the current sheet
    pred_loc = np.array([[0, 0, R_EARTH + 1e6]])
    J_pred = secs.predict(pred_loc, J=True)
    expected = np.array([0, 0, 1.169507e-14])
    assert_allclose(expected, J_pred, rtol=1e-6, atol=1e-10)

    # Use the predict_J function call directly
    assert_allclose(secs.predict_J(pred_loc), secs.predict(pred_loc, J=True))


def test_multidim_shapes():
    """Test multidimensional prediction."""
    np.random.seed(0)
    nsec = 100
    nobs = 10
    ntimes = 75
    npred = 133
    sec_locs = np.random.rand(nsec, 3) * 100
    obs_locs = np.random.rand(nobs, 3) * 100
    obs_B = np.random.rand(ntimes, nobs, 3) * 10000
    pred_locs = np.random.rand(npred, 3) * 100

    secs = pysecs.SECS(sec_df_loc=sec_locs)
    assert secs.nsec == nsec

    secs.fit(obs_locs, obs_B=obs_B)
    assert secs.sec_amps.shape == (ntimes, nsec)
    assert secs.sec_amps_var.shape == (ntimes, nsec)

    pred = secs.predict(pred_locs)
    assert pred.shape == (ntimes, npred, 3)


def test_changing_obs_shape():
    # If we change the shape of the obs data, we don't want to
    # cache the old data and use that, we want to recompute with a new shape
    np.random.seed(0)
    nsec = 100
    nobs = 10
    ntimes = 75
    npred = 133
    sec_locs = np.random.rand(nsec, 3) * 100
    obs_locs = np.random.rand(nobs, 3) * 100
    obs_B = np.random.rand(ntimes, nobs, 3) * 10000
    pred_locs = np.random.rand(npred, 3) * 100

    secs = pysecs.SECS(sec_df_loc=sec_locs)

    secs.fit(obs_locs, obs_B=obs_B)
    # Now change the shape of the obs data by removing the last observation
    secs.fit(obs_locs[:-1], obs_B=obs_B[:, :-1, :])
    assert secs.sec_amps.shape == (ntimes, nsec)
    assert secs.sec_amps_var.shape == (ntimes, nsec)

    pred = secs.predict(pred_locs)
    assert pred.shape == (ntimes, npred, 3)
    pred = secs.predict(pred_locs[:-1])
    assert pred.shape == (ntimes, npred - 1, 3)
