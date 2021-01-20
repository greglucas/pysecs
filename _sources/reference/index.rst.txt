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
    SECS.predict
    SECS.predict_B
    SECS.predict_J

    SECS.has_df
    SECS.has_cf
    SECS.nsec

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