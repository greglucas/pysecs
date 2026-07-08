.. _theory:

Theory
======

This page collects the equations implemented in pysecs, their sign and
coordinate conventions, and the derivation of the curl-free magnetic
field that is stated but not written out in the original papers.

Elementary current systems
--------------------------

Spherical elementary current systems (SECS) were introduced by
Amm (1997) [1]_ as a pair of basis functions for two-dimensional vector
fields on a sphere. In a spherical coordinate system
:math:`(r, \theta', \phi')` whose pole :math:`\theta' = 0` is at the
SEC location ("pole") on a shell of radius :math:`R`, the
divergence-free (df) and curl-free (cf) sheet current densities are

.. math::

    \vec{J}_{df}(\theta') = \frac{I_0}{4 \pi R} \cot(\theta'/2)\,
    \hat{e}_{\phi'},
    \qquad
    \vec{J}_{cf}(\theta') = \frac{I_0}{4 \pi R} \cot(\theta'/2)\,
    \hat{e}_{\theta'},

(Amm & Viljanen 1999 [2]_, Eqs. 6 and 7). The scaling factors
:math:`I_0` (in Amperes) are the free parameters fit to observations.

The curl-free system is closed by radial field-aligned currents (FACs):
a line current :math:`I_0` flowing *into* the ionosphere at the pole and
uniformly distributed FACs of density :math:`I_0 / (4 \pi r^2)` flowing
*out* everywhere else, so the net FAC over the shell is zero. A positive
amplitude therefore means: FAC into the shell at the pole, sheet current
away from the pole, distributed return FACs outward.

Magnetic field of the divergence-free system
--------------------------------------------

Amm & Viljanen (1999) derived the magnetic field of the df system from
its vector potential. Below the shell (:math:`r < R`, their Eqs. 9 and
10):

.. math::

    B_r(r, \theta') = \frac{\mu_0 I_0}{4 \pi r}
    \left( \frac{1}{\sqrt{1 - 2 s \cos\theta' + s^2}} - 1 \right),
    \qquad s = r / R,

.. math::

    B_{\theta'}(r, \theta') = -\frac{\mu_0 I_0}{4 \pi r \sin\theta'}
    \left( \frac{s - \cos\theta'}{\sqrt{1 - 2 s \cos\theta' + s^2}}
    + \cos\theta' \right),

with the corresponding expressions above the shell given in their
Appendix (Eqs. A.7 and A.8). These are implemented in
:func:`pysecs.T_df`.

Magnetic field of the curl-free system
--------------------------------------

Amm (1997) and Amm & Viljanen (1999) state, but do not derive, that the
cf system (with its FACs) produces **no magnetic field at or below the
current shell** -- the spherical generalization of Fukushima's theorem --
and a purely toroidal field above it. The explicit field was later
published as Eq. 2.15 of Vanhamäki & Juusola (2020) [3]_ and is
implemented in :func:`pysecs.T_cf`:

.. math::

    \vec{B}_{cf}(r, \theta') =
    \begin{cases}
        0, & r < R, \\[4pt]
        -\dfrac{\mu_0 I_0}{4 \pi r} \cot(\theta'/2)\, \hat{e}_{\phi'},
        & r > R.
    \end{cases}

The derivation only needs Ampère's law. The current system is
axisymmetric and purely poloidal (only :math:`r` and :math:`\theta`
components), so :math:`\vec{B}` is purely toroidal,
:math:`\vec{B} = B_{\phi'} \hat{e}_{\phi'}`. Integrating around a circle
of constant :math:`(r, \theta')`:

- For :math:`r < R` no current threads the enclosed polar cap (the sheet
  and all FACs lie above), so :math:`B_{\phi'} = 0` exactly.
- For :math:`r > R` the cap is threaded by the pole line current
  :math:`-I_0` (downward) and the distributed return current
  :math:`+I_0 (1 - \cos\theta')/2`, giving
  :math:`2 \pi r \sin\theta' B_{\phi'} =
  -\mu_0 I_0 (1 + \cos\theta')/2`, i.e. the expression above using
  :math:`(1 + \cos\theta)/\sin\theta = \cot(\theta/2)`.

The result satisfies four independent checks (all verified numerically
in ``tests/test_physics.py``):

1. as :math:`\theta' \to 0` it approaches the field
   :math:`\mu_0 I_0 / (2 \pi \rho)` of an infinite straight line
   current with the correct handedness;
2. the jump across the shell equals
   :math:`\mu_0 (\vec{K} \times \hat{n})` for the sheet current of
   Eq. 7;
3. :math:`\nabla \times \vec{B} = \mu_0 I_0 / (4 \pi r^2)\, \hat{e}_r`
   above the shell, exactly the distributed FAC density;
4. :math:`\nabla \cdot \vec{B} = 0`.

.. important::

    Because the cf field vanishes at and below the shell, **ground
    observations cannot constrain curl-free amplitudes**. Fitting cf
    systems requires observations above the current shell (e.g.
    satellite magnetometers). :meth:`pysecs.SECS.fit` warns when cf
    SECs cannot be constrained by the supplied observation geometry.

    The zero-below-shell result assumes radial FACs, which is a good
    approximation at high magnetic latitudes but degrades equatorward
    of roughly 60 degrees magnetic latitude where the field lines are
    inclined (Tamao 1986; see the discussion in [3]_).

Coordinate conventions
----------------------

All user-facing locations are ``(latitude [deg], longitude [deg],
radius [m])``. Vector components are geographic
``(X north, Y east, Z down)`` at each location, matching ground
magnetometer conventions. Internally the SEC-centered components
:math:`(\hat{e}_{r}, \hat{e}_{\theta'}, \hat{e}_{\phi'})` are rotated
using the bearing between each observation and SEC pole.

Fitting and uncertainty
-----------------------

:meth:`pysecs.SECS.fit` solves :math:`T I = Z` in the least-squares
sense using the SVD of the (uncertainty-weighted) transfer matrix, with
small singular values truncated by ``epsilon`` (Amm & Viljanen 1999,
Sect. 5). Observation weighting follows from the 1-sigma standard
errors ``obs_std``; infinite values eliminate observations.

With :math:`A = V W U^T` the truncated inverse operator and scaled
observations of unit variance, the amplitude covariance is
:math:`A A^T`; its diagonal is stored in ``sec_amps_var`` and the full
covariance is propagated through the prediction transfer matrix when
``predict(..., return_var=True)`` is requested.

Robust fitting (``fit(..., robust='huber'|'bisquare')``) applies
iteratively reweighted least squares: standardized residuals are scaled
by their per-time-step normalized median absolute deviation and mapped
to weights so that observations inconsistent with any smooth current
system (single-station spikes, baseline jumps) are automatically
downweighted.

Temporal estimation
--------------------

:class:`pysecs.KalmanSECS` couples the time steps with a
linear-Gaussian state-space model on the amplitudes,

.. math::

    m_t = \phi_t\, m_{t-1} + w_t, \qquad
    \phi_t = e^{-\Delta t / \tau}, \qquad
    w_t \sim N\!\left(0,\ (1 - \phi_t^2)\, \sigma_p^2 I\right),

an exactly discretized Ornstein-Uhlenbeck process with correlation time
:math:`\tau` and stationary standard deviation :math:`\sigma_p`
(``prior_std``), observed through the SECS transfer matrix. A Kalman
filter (optionally followed by a Rauch-Tung-Striebel smoother) then
estimates the amplitudes and their covariance at every step. Innovation
gating (``gate_sigma``) inflates the errors of observations that are
inconsistent with the forecast, rejecting impulsive disturbances while
following consistent large-scale changes. See Laundal et al. (2025)
[4]_ for a review of temporal regularization in ionospheric data
assimilation.

References
----------

.. [1] Amm, O. "Ionospheric Elementary Current Systems in Spherical
    Coordinates and Their Application." J. Geomag. Geoelectr. 49
    (1997): 947-955. doi:10.5636/jgg.49.947
.. [2] Amm, O., and A. Viljanen. "Ionospheric disturbance magnetic
    field continuation from the ground to the ionosphere using
    spherical elementary current systems." Earth, Planets and Space
    51 (1999): 431-440. doi:10.1186/BF03352247
.. [3] Vanhamäki, H., and L. Juusola. "Introduction to Spherical
    Elementary Current Systems." Ionospheric Multi-Spacecraft Analysis
    Tools, ISSI Scientific Report Series 17 (2020): 5-33.
    doi:10.1007/978-3-030-26732-2_2
.. [4] Laundal, K. M., et al. "Next-Generation Data Assimilation
    Methods for Polar Ionospheric Electrodynamics." Surveys in
    Geophysics (2025). doi:10.1007/s10712-025-09918-3
