.. _api:

API reference
=============
.. currentmodule:: pysecs

This page gives an overview of all public classes and functions
within the pysecs package.


.. rubric:: SECS Class

The primary class to set up a Spherical Elementary Current System is
the SECS class.

.. autosummary::
    :toctree: generated/
    
    SECS


.. rubric:: SECS Methods

The `fit()` and `predict()` methods can be called on the created system
to fit to data and then predict at any other location.

.. autosummary::
    :toctree: generated/
    
    SECS.fit
    SECS.fit_unit_currents
    SECS.from_observations
    SECS.predict
    SECS.predict_B
    SECS.predict_J

    SECS.has_df
    SECS.has_cf
    SECS.nsec

.. rubric:: Temporal estimation

The KalmanSECS class couples time steps together with a state-space
model on the SEC amplitudes, estimated with a Kalman filter and
Rauch-Tung-Striebel smoother.

.. autosummary::
    :toctree: generated/

    KalmanSECS
    KalmanSECS.fit
    KalmanSECS.predict
    KalmanSECS.predict_B
    KalmanSECS.predict_J

.. rubric:: Grid generation

``SECS.from_observations`` builds a grid automatically with
``make_grid``, which can also be called directly for more control.

.. autosummary::
    :toctree: generated/

    make_grid
    make_image_grid

.. rubric:: Additional functions

The helper functions create the geometrical transforms going from
a pole to observation point are described below.

.. autosummary::
    :toctree: generated/

    T_df
    T_cf
    J_df
    J_cf
    calc_angular_distance
    calc_bearing
