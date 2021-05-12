"""
Fitting B time series (animation)
---------------------------------

This example demonstrates how to fit generic
B observation inputs and fit an SECS system
to make predictions on a separate grid and
compare the results.
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from pysecs import SECS


R_earth = 6371e3

# specify the SECS grid
lat, lon, r = np.meshgrid(np.linspace(-20, 20, 30),
                          np.linspace(-20, 20, 30),
                          R_earth + 110000, indexing='ij')
secs_lat_lon_r = np.hstack((lat.reshape(-1, 1),
                            lon.reshape(-1, 1),
                            r.reshape(-1, 1)))

# Set up the class
secs = SECS(sec_df_loc=secs_lat_lon_r)

# Make a grid of input observations spanning
# (-10, 10) in latitutde and longitude
lat, lon, r = np.meshgrid(np.linspace(-10, 10, 11),
                          np.linspace(-10, 10, 11),
                          R_earth, indexing='ij')
obs_lat = lat[..., 0]
obs_lon = lon[..., 0]
obs_lat_lon_r = np.hstack((lat.reshape(-1, 1),
                           lon.reshape(-1, 1),
                           r.reshape(-1, 1)))
nobs = len(obs_lat_lon_r)

# Create the synthetic magnetic field data as a function
# of time
ts = np.linspace(0, 2*np.pi)
bx = 5*np.cos(ts)
by = 5*np.sin(ts)
bz = ts
# ntimes x 3
B_obs = np.column_stack([bx, by, bz])
ntimes = len(B_obs)

# Repeat that for each observatory
# ntimes x nobs x 3
B_obs = np.repeat(B_obs[:, np.newaxis, :], nobs, axis=1)
# Make it more interesting and add a sin wave in spatial
# coordinates too
B_obs[:, :, 0] *= 2*np.sin(np.deg2rad(obs_lat_lon_r[:, 0]))
B_obs[:, :, 1] *= 2*np.sin(np.deg2rad(obs_lat_lon_r[:, 1]))

B_std = np.ones(B_obs.shape)
# Ignore the Z component
B_std[..., 2] = np.inf
# Can modify the standard error as a function of time to
# see how that changes the fits too
# B_std[:, 0, 1] = 1 + ts

# Fit the data, requires observation locations and data
secs.fit(obs_loc=obs_lat_lon_r, obs_B=B_obs, obs_std=B_std)

# Create prediction points
# Extend it a little beyond the observation points (-11, 11)
lat, lon, r = np.meshgrid(np.linspace(-11, 11, 11),
                          np.linspace(-11, 11, 11),
                          R_earth, indexing='ij')
pred_lat_lon_r = np.hstack((lat.reshape(-1, 1),
                            lon.reshape(-1, 1),
                            r.reshape(-1, 1)))

# Call the prediction function
B_pred = secs.predict(pred_lat_lon_r)

# Now set up the plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                             sharex=True, sharey=True)
vmin, vmax = -5, 5
cmap = plt.get_cmap('RdBu_r')
t = 10

mesh1 = ax1.pcolormesh(obs_lon, obs_lat, B_obs[t, :, 0].reshape(obs_lon.shape),
                       vmin=vmin, vmax=vmax, cmap=cmap, shading='auto')
mesh2 = ax2.pcolormesh(obs_lon, obs_lat, B_obs[t, :, 1].reshape(obs_lon.shape),
                       vmin=vmin, vmax=vmax, cmap=cmap, shading='auto')
ax1.set_ylabel("Observations")
ax1.set_title("B$_{X}$")
ax2.set_title("B$_{Y}$")

qscale = 1
q1 = ax1.quiver(obs_lat_lon_r[:, 1], obs_lat_lon_r[:, 0],
                B_obs[t, :, 1], B_obs[t, :, 0],
                angles='xy', scale_units='xy', scale=qscale)
q2 = ax2.quiver(obs_lat_lon_r[:, 1], obs_lat_lon_r[:, 0],
                B_obs[t, :, 1], B_obs[t, :, 0],
                angles='xy', scale_units='xy', scale=qscale)

lon = lon[..., 0]
lat = lat[..., 0]
mesh3 = ax3.pcolormesh(lon, lat, B_pred[t, :, 0].reshape(lon.shape),
                       vmin=vmin, vmax=vmax, cmap=cmap, shading='auto')
mesh4 = ax4.pcolormesh(lon, lat, B_pred[t, :, 1].reshape(lon.shape),
                       vmin=vmin, vmax=vmax, cmap=cmap, shading='auto')

q3 = ax3.quiver(pred_lat_lon_r[:, 1], pred_lat_lon_r[:, 0],
                B_pred[t, :, 1], B_pred[t, :, 0],
                angles='xy', scale_units='xy', scale=qscale)
q4 = ax4.quiver(pred_lat_lon_r[:, 1], pred_lat_lon_r[:, 0],
                B_pred[t, :, 1], B_pred[t, :, 0],
                angles='xy', scale_units='xy', scale=qscale)
ax3.set_ylabel("Predictions")
ax3.set_title("B$_{X}$")
ax4.set_title("B$_{Y}$")


def update_axes(t):
    # Update the mesh colors
    mesh1.set_array(B_obs[t, :, 0].reshape(obs_lon.shape))
    mesh2.set_array(B_obs[t, :, 1].reshape(obs_lon.shape))
    mesh3.set_array(B_pred[t, :, 0].reshape(lon.shape))
    mesh4.set_array(B_pred[t, :, 1].reshape(lon.shape))

    # Update the quiver arrows
    q1.set_UVC(B_obs[t, :, 1], B_obs[t, :, 0])
    q2.set_UVC(B_obs[t, :, 1], B_obs[t, :, 0])
    q3.set_UVC(B_pred[t, :, 1], B_pred[t, :, 0])
    q4.set_UVC(B_pred[t, :, 1], B_pred[t, :, 0])


ani = FuncAnimation(fig, update_axes, frames=range(ntimes),
                    interval=50)

plt.show()
