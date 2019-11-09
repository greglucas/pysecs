import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest
import pySECS

R_EARTH = 6378e3


def test_angular_distance():
    "Test the angular distance formula."
    latlon1 = np.array([[0., 0.]])
    latlon2 = np.array([[0., 0.], [0., 90], [-90., 0.], [0., 180.]])
    assert_array_equal(pySECS.calc_angular_distance(latlon1, latlon2),
                       np.deg2rad([[0., 90., 90., 180.]]))


def test_bearing():
    "Test the cardinal directions."
    latlon1 = np.array([[0., 0.]])
    latlon2 = np.array([[0., 90.], [90., 0.], [90., 45.], [0., -90.], [-90, 0.]])
    assert_array_equal(pySECS.calc_bearing(latlon1, latlon2),
                       np.deg2rad([[0., 90., 90., 180., -90]]))


def test_divergence_free_magnetic_directions():
    "Make sure the divergence free magnetic field angles are correct"
    # Place the SEC at the equator
    sec_r = R_EARTH + 100
    sec_latlonr = np.array([[0., 0., sec_r]])
    # Going around in a circle from the point
    obs_latlonr = np.array([[5., 0., R_EARTH], [0., 5., R_EARTH],
                            [-5, 0., R_EARTH], [0., -5., R_EARTH]])

    B = np.squeeze(pySECS.T_df(obs_latlonr, sec_latlonr))

    angles = np.arctan2(B[:, 0], B[:, 1])
    # southward, westward, northward, eastward
    expected_angles = np.deg2rad([-90, 180., 90., 0.])
    assert_allclose(angles, expected_angles, rtol=1e-10, atol=1e-10)


def test_divergence_free_magnetic_magnitudes_obs_under():
    "Make sure the divergence free magnetic amplitudes are correct."
    # Place the SEC at the North Pole
    sec_r = R_EARTH + 100
    sec_latlonr = np.array([[0., 0., sec_r]])
    # Going out in an angle from the SEC (in longitude)
    angles = np.linspace(0.1, 180)
    obs_r = R_EARTH
    obs_latlonr = np.zeros(angles.shape + (3,))
    obs_latlonr[:, 1] = angles
    obs_latlonr[:, 2] = obs_r

    B = np.squeeze(pySECS.T_df(obs_latlonr, sec_latlonr))

    # All x components should be zero (angles goes around the equator and all
    # quantities should be parallel to that)
    assert_allclose(np.zeros(angles.shape), B[:, 0], atol=1e-16)

    # Actual magnitude
    mu0 = 4*np.pi*1e-7

    # simplify calculations by storing this ratio
    x = obs_r/sec_r

    sin_theta = np.sin(np.deg2rad(angles))
    cos_theta = np.cos(np.deg2rad(angles))
    factor = 1./np.sqrt(1 - 2*x*cos_theta + x**2)

    # Amm & Viljanen: Equation 9
    Br = mu0/(4*np.pi*obs_r) * (factor - 1)
    # Bz in opposite direction of Br
    assert_allclose(-Br, B[:, 2])

    # Amm & Viljanen: Equation 10
    Btheta = -mu0/(4*np.pi*obs_r) * (factor*(x - cos_theta) + cos_theta)
    Btheta = np.divide(Btheta, sin_theta, out=np.zeros_like(sin_theta), where=sin_theta != 0)
    assert_allclose(Btheta, B[:, 1])


def test_divergence_free_magnetic_magnitudes_obs_over():
    "Make sure the divergence free magnetic amplitudes are correct."
    # Place the SEC at the North Pole
    sec_r = R_EARTH
    sec_latlonr = np.array([[0., 0., sec_r]])
    # Going out in an angle from the SEC (in longitude)
    angles = np.linspace(0.1, 180)
    obs_r = R_EARTH + 100
    obs_latlonr = np.zeros(angles.shape + (3,))
    obs_latlonr[:, 1] = angles
    obs_latlonr[:, 2] = obs_r

    B = np.squeeze(pySECS.T_df(obs_latlonr, sec_latlonr))

    # All x components should be zero (angles goes around the equator and all
    # quantities should be parallel to that)
    assert_allclose(np.zeros(angles.shape), B[:, 0], atol=1e-16)

    # Actual magnitude
    mu0 = 4*np.pi*1e-7
    x = sec_r/obs_r

    sin_theta = np.sin(np.deg2rad(angles))
    cos_theta = np.cos(np.deg2rad(angles))

    # Amm & Viljanen: Equation A.7
    Br = mu0*x/(4*np.pi*obs_r) * (1./np.sqrt(1 - 2*x*cos_theta + x**2) - 1)
    # Bz in opposite direction of Br
    assert_allclose(-Br, B[:, 2])

    # Amm & Viljanen: Equation A.8
    Btheta = -mu0/(4*np.pi*obs_r)*((obs_r-sec_r*cos_theta) /
                                   np.sqrt(obs_r**2 - 2*obs_r*sec_r*cos_theta + sec_r**2) - 1)
    Btheta = np.divide(Btheta, sin_theta, out=np.zeros_like(sin_theta), where=sin_theta != 0)
    assert_allclose(Btheta, B[:, 1])


def test_outside_current_plane():
    "Make sure all currents outside the SEC plane are 0."
    sec_r = R_EARTH + 100
    sec_latlonr = np.array([[0., 0., sec_r]])
    # Above and below the plane, also on and off the SEC point
    obs_latlonr = np.array([[0., 0., sec_r - 100.], [0., 0., sec_r + 100.],
                            [5, 0., sec_r - 100.], [5., 0., sec_r + 100.]])

    # df currents
    J = np.squeeze(pySECS.J_df(obs_latlonr, sec_latlonr))
    assert np.all(J == 0.)
    # cf currents
    J = np.squeeze(pySECS.J_cf(obs_latlonr, sec_latlonr))
    assert np.all(J == 0.)


def test_divergence_free_current_directions():
    "Make sure the divergence free current angles are correct."
    # Place the SEC at the equator
    sec_r = R_EARTH + 100
    sec_latlonr = np.array([[0., 0., sec_r]])
    # Going around in a circle from the point
    obs_latlonr = np.array([[5., 0., sec_r], [0., 5., sec_r],
                            [-5, 0., sec_r], [0., -5., sec_r]])

    J = np.squeeze(pySECS.J_df(obs_latlonr, sec_latlonr))

    angles = np.arctan2(J[:, 0], J[:, 1])
    # westward, northward, eastward, southward
    expected_angles = np.deg2rad([-180., 90., 0., -90.])
    assert_allclose(angles, expected_angles, atol=1e-16)


def test_divergence_free_current_magnitudes():
    "Make sure the divergence free current amplitudes are correct."
    # Place the SEC at the North Pole
    sec_r = R_EARTH + 100
    sec_latlonr = np.array([[0., 0., sec_r]])
    # Going out in an angle from the SEC (in longitude)
    angles = np.linspace(0.1, 180)
    obs_latlonr = np.zeros(angles.shape + (3,))
    obs_latlonr[:, 1] = angles
    obs_latlonr[:, 2] = sec_r

    J = np.squeeze(pySECS.J_df(obs_latlonr, sec_latlonr))

    # Make sure all radial components are zero in this system
    assert np.all(J[:, 2] == 0.)

    # Also all y components (angles goes around the equator and all
    # quantities should be perpendicular to that)
    assert_allclose(np.zeros(angles.shape), J[:, 1], atol=1e-16)

    # Actual magnitude
    tan_theta2 = np.tan(np.deg2rad(angles/2))
    J_test = 1./(4*np.pi*sec_r)
    J_test = np.divide(J_test, tan_theta2, out=np.ones_like(tan_theta2)*np.inf,
                       where=tan_theta2 != 0.)

    assert_allclose(J_test, J[:, 0], atol=1e-16)


def test_curl_free_current_directions():
    "Make sure the curl free current angles are correct."
    # Place the SEC at the equator
    sec_r = R_EARTH + 100
    sec_latlonr = np.array([[0., 0., sec_r]])
    # Going around in a circle from the point
    obs_latlonr = np.array([[5., 0., sec_r], [0., 5., sec_r],
                            [-5, 0., sec_r], [0., -5., sec_r]])

    J = np.squeeze(pySECS.J_cf(obs_latlonr, sec_latlonr))

    angles = np.arctan2(J[:, 0], J[:, 1])
    # pointing out from the SEC direction to OBS direction.
    # northward, eastward, southward, westward
    expected_angles = np.deg2rad([90., 0., -90., -180])
    assert_allclose(angles, expected_angles, atol=1e-15)


def test_curl_free_current_magnitudes():
    "Make sure the curl free current amplitudes are correct."
    # Place the SEC at the North Pole
    sec_r = R_EARTH + 100
    sec_latlonr = np.array([[0., 0., sec_r]])
    # Going out in an angle from the SEC (in longitude)
    angles = np.linspace(0.1, 180)
    obs_latlonr = np.zeros(angles.shape + (3,))
    obs_latlonr[:, 1] = angles
    obs_latlonr[:, 2] = sec_r

    J = np.squeeze(pySECS.J_cf(obs_latlonr, sec_latlonr))

    # Make sure all radial components are oppositely directed
    radial_component = 1./(4*np.pi*sec_r**2)
    assert np.all(J[:, 2] == radial_component)

    # All x components should be zero (angles goes around the equator and all
    # quantities should be parallel to that)
    # (ambiguous 0 degree angle so ignore the first input)
    assert_allclose(np.zeros(angles.shape), J[:, 0], atol=1e-16)

    # Actual magnitude
    tan_theta2 = np.tan(np.deg2rad(angles/2))
    J_test = 1./(4*np.pi*sec_r)
    J_test = np.divide(J_test, tan_theta2, out=np.ones_like(tan_theta2)*np.inf,
                       where=tan_theta2 != 0.)

    assert_allclose(J_test, J[:, 1], atol=1e-16)


def test_empty_object():
    "Testing empty secs object creation failure."
    with pytest.raises(ValueError):
        pySECS.SECS()


def test_list_numpy():
    "Make sure creation with numpy and list produce the same locations."
    x2d = [[1., 0., 0.], [1., 0., 0.]]
    x2d_np = np.array(x2d)
    secs_list2 = pySECS.SECS(sec_df_loc=x2d, sec_cf_loc=x2d)
    secs_np2 = pySECS.SECS(sec_df_loc=x2d_np, sec_cf_loc=x2d_np)
    assert secs_list2.nsec == 4  # 2 df + 2 cf
    assert secs_list2.nsec == secs_np2.nsec
    assert_array_equal(secs_list2.sec_df_loc, secs_np2.sec_df_loc)
    assert_array_equal(secs_list2.sec_cf_loc, secs_np2.sec_cf_loc)
