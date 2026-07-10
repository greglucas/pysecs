"""
Choosing epsilon with cross-validation
--------------------------------------

The ``epsilon`` parameter of :meth:`pysecs.SECS.fit` controls how many
singular values are kept in the inversion: too small and the fit chases
noise, too large and real structure is smoothed away. Leave-one-out
cross-validation gives an objective way to choose it: hold out one
station, fit the rest, and measure the prediction error at the held-out
station.
"""

import matplotlib.pyplot as plt
import numpy as np

from pysecs import SECS, T_df


R_E = 6378e3
rng = np.random.default_rng(42)

# A grid of divergence-free SECs at 110 km altitude
lats = np.linspace(35, 55, 8)
lons = np.linspace(-10, 20, 10)
lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
sec_locs = np.column_stack(
    [lat_g.ravel(), lon_g.ravel(), np.full(lat_g.size, R_E + 110e3)]
)

# Synthetic "true" current system and noisy ground observations
true_amps = rng.normal(0, 1e4, sec_locs.shape[0])
nobs = 25
obs_locs = np.column_stack(
    [rng.uniform(36, 54, nobs), rng.uniform(-9, 19, nobs), np.full(nobs, R_E)]
)
B_true = np.tensordot(true_amps, T_df(obs_locs, sec_locs), (0, 2))
noise = 0.05 * np.max(np.abs(B_true))
B_obs = B_true + rng.normal(0, noise, B_true.shape)

# Leave-one-out cross-validation over a range of epsilons
epsilons = np.logspace(-4, 0, 20)
cv_error = np.zeros_like(epsilons)
for i, epsilon in enumerate(epsilons):
    for k in range(nobs):
        keep = np.arange(nobs) != k
        secs = SECS(sec_df_loc=sec_locs)
        secs.fit(obs_locs[keep], B_obs[keep], epsilon=epsilon)
        pred = secs.predict_B(obs_locs[[k]])
        cv_error[i] += np.sum((pred - B_obs[k]) ** 2)
cv_error = np.sqrt(cv_error / (nobs * 3))

best = epsilons[np.argmin(cv_error)]

fig, ax = plt.subplots()
ax.loglog(epsilons, cv_error * 1e9, marker="o")
ax.axvline(best, color="tab:red", ls="--", label=f"best epsilon = {best:.2g}")
ax.axhline(noise * 1e9, color="gray", ls=":", label="noise level")
ax.set_xlabel("epsilon")
ax.set_ylabel("leave-one-out RMSE (nT)")
ax.set_title("Cross-validated choice of the regularization parameter")
ax.legend()
plt.show()
