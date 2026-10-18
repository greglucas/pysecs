"""
Fitting currents and joint magnetic field + current fits
---------------------------------------------------------

Ground magnetometers only see the divergence-free part of the
ionospheric currents (the "equivalent currents"): the curl-free part,
together with its field-aligned currents (FACs), produces no magnetic
field below the ionosphere. Estimates of the horizontal sheet current
itself -- for example from radar electric fields and conductances --
see both parts.

This example builds a synthetic electrojet with a pair of FAC sheets
and compares three fits of a SECS with divergence-free and curl-free
poles:

- :meth:`pysecs.SECS.fit` to ground magnetometers only, which recovers
  the equivalent currents but no FACs,
- :meth:`pysecs.SECS.fit_J` to currents in a limited field of view,
  which recovers the FACs only where the currents are observed,
- a joint fit to both with ``fit(..., J_loc=, obs_J=, J_std=)``, which
  combines the wide ground coverage with the FAC information.

The FAC density is the curl-free amplitude of each SEC divided by the
area of its grid cell.
"""

import matplotlib.pyplot as plt
import numpy as np

from pysecs import SECS


R_E = 6378e3
R_I = R_E + 110e3
rng = np.random.default_rng(42)


def regular_grid(lats, lons, r):
    """Regular lat/lon grid flattened to (n, 3 [lat, lon, r])."""
    lat, lon = np.meshgrid(lats, lons, indexing="ij")
    return np.column_stack([lat.ravel(), lon.ravel(), np.full(lat.size, r)])


def fac_density(secs, dlat, dlon):
    """Downward FAC density (A/m^2) from the curl-free amplitudes."""
    lat = np.deg2rad(secs.sec_cf_loc[:, 0])
    area = R_I**2 * np.deg2rad(dlat) * np.deg2rad(dlon) * np.cos(lat)
    return secs.sec_amps[0, len(secs.sec_df_loc) :] / area


# %%
# Synthetic truth
# ^^^^^^^^^^^^^^^
# A westward electrojet (divergence-free) between a downward and an
# upward FAC sheet (curl-free), on a finer grid than the one we fit.
true_dlat, true_dlon = 1.0, 2.0
true_grid = regular_grid(
    np.arange(56.25, 80, true_dlat), np.arange(-39.5, 50, true_dlon), R_I
)
lat, lon = true_grid[:, 0], true_grid[:, 1]
envelope = np.exp(-(((lon - 5) / 18) ** 2))
# Amplitudes changing sign across the jet give a band of east-west current
x = (lat - 67) / 3.5
true_df = -5e4 * x * np.exp(-(x**2)) * envelope
true_cf = 8e3 * (
    np.exp(-(((lat - 70) / 1.2) ** 2)) - np.exp(-(((lat - 64) / 1.2) ** 2))
)
true_cf *= envelope

truth = SECS(sec_df_loc=true_grid, sec_cf_loc=true_grid)
truth.sec_amps = np.concatenate([true_df, true_cf])[np.newaxis]

# %%
# Observations
# ^^^^^^^^^^^^
# Ground magnetometers spread over the whole region, and noisy sheet
# current estimates on the ionospheric shell within a smaller field of
# view. Currents are only defined on the SEC shell, so their radius must
# equal the shell radius.
nmag = 40
mag_loc = np.column_stack(
    [rng.uniform(58, 76, nmag), rng.uniform(-25, 35, nmag), np.full(nmag, R_E)]
)
B_std = np.full((nmag, 3), 5e-9)  # 5 nT
B_obs = truth.predict_B(mag_loc) + rng.normal(0, B_std)

J_loc = regular_grid(np.arange(62.5, 73, 1.0), np.arange(-7.5, 18, 2.0), R_I)
J_std = np.full((len(J_loc), 2), 0.05)  # 50 mA/m
J_obs = truth.predict_J(J_loc)[:, :2] + rng.normal(0, J_std)

# %%
# Fits
# ^^^^
# All fits use the same grid of divergence-free and curl-free SECs. The
# ground-only fit uses divergence-free SECs alone, since the curl-free
# amplitudes cannot be determined from the ground. The joint fit
# weights the two kinds of observations by their standard errors, which
# are required because they have different units.
dlat, dlon = 2.0, 4.0
grid = regular_grid(np.arange(56, 80, dlat), np.arange(-40, 50, dlon), R_I)

ground = SECS(sec_df_loc=grid).fit(mag_loc, B_obs, B_std, epsilon=0.05)

currents = SECS(sec_df_loc=grid, sec_cf_loc=grid)
currents.fit_J(J_loc, J_obs, J_std, epsilon=0.05)

joint = SECS(sec_df_loc=grid, sec_cf_loc=grid)
joint.fit(mag_loc, B_obs, B_std, epsilon=0.05, J_loc=J_loc, obs_J=J_obs, J_std=J_std)

# %%
# Comparison
# ^^^^^^^^^^
# Horizontal currents are evaluated between the grid poles, where each
# SEC's current is singular, and the FAC density is shown on the grid.
plot_loc = regular_grid(np.arange(59, 77, 2.0), np.arange(-28, 38, 4.0), R_I)

panels = [
    ("Truth", truth, fac_density(truth, true_dlat, true_dlon)),
    ("Ground magnetometers only", ground, None),
    ("Currents only (fit_J)", currents, fac_density(currents, dlat, dlon)),
    ("Joint fit", joint, fac_density(joint, dlat, dlon)),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
vmax = 1.0  # uA/m^2
for ax, (title, secs, fac) in zip(axes.flat, panels, strict=True):
    if fac is not None:
        loc = secs.sec_cf_loc
        im = ax.scatter(
            loc[:, 1],
            loc[:, 0],
            c=fac * 1e6,
            s=12 if secs is truth else 50,
            marker="s",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
    J = secs.predict_J(plot_loc)
    # Arrows point (east, north) with plot x = longitude and y = latitude
    ax.quiver(plot_loc[:, 1], plot_loc[:, 0], J[:, 1], J[:, 0], scale=5)
    ax.plot(mag_loc[:, 1], mag_loc[:, 0], "k^", ms=4, label="magnetometers")
    ax.add_patch(
        plt.Rectangle(
            (-7.5, 62.5),
            25,
            10,
            fill=False,
            ec="C2",
            lw=2,
            label="current observations",
        )
    )
    ax.set_title(title)
    ax.set_xlim(-30, 40)
    ax.set_ylim(57, 78)

axes[0, 1].text(
    5, 75, "no FAC information", ha="center", va="center", backgroundcolor="w"
)
for ax in axes[1]:
    ax.set_xlabel("longitude (deg)")
for ax in axes[:, 0]:
    ax.set_ylabel("latitude (deg)")
axes[0, 0].legend(loc="lower left", fontsize="small")
fig.colorbar(im, ax=axes, label=r"downward FAC ($\mu$A/m$^2$)", shrink=0.8)
plt.show()
