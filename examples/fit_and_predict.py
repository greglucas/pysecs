import matplotlib.pyplot as plt
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

# 4 input observatories around the equator
obs_lat_lon_r = np.array([[-5, -5, R_earth],
                          [-5, 5, R_earth],
                          [5, 5, R_earth],
                          [5, -5, R_earth]])
lat, lon, r = np.meshgrid(np.linspace(-10, 10, 11),
                          np.linspace(-10, 10, 11),
                          R_earth, indexing='ij')
obs_lat = lat[..., 0]
obs_lon = lon[..., 0]
obs_lat_lon_r = np.hstack((lat.reshape(-1, 1),
                           lon.reshape(-1, 1),
                           r.reshape(-1, 1)))

nobs = len(obs_lat_lon_r)

# Set up the magnetic field data
ts = np.linspace(0, 2*np.pi)
bx = 5*np.cos(ts)
by = 5*np.sin(ts)
bz = ts
# ntimes x 3
B_obs = np.column_stack([bx, by, bz])
# ntimes x nobs x 3
B_obs = np.repeat(B_obs[:, np.newaxis, :], nobs, axis=1)
B_obs[:, :, 0] *= 2*np.sin(np.deg2rad(obs_lat_lon_r[:, 0]))
B_obs[:, :, 1] *= 2*np.sin(np.deg2rad(obs_lat_lon_r[:, 1]))

B_var = np.ones(B_obs.shape)
# Ignore the Z component
B_var[..., 2] = np.inf
# B_var[:, 0, 1] = 1 + ts

secs.fit(obs_loc=obs_lat_lon_r, obs_B=B_obs, obs_var=B_var)

# Create prediction points
lat, lon, r = np.meshgrid(np.linspace(-11, 11, 11),
                          np.linspace(-11, 11, 11),
                          R_earth, indexing='ij')
pred_lat_lon_r = np.hstack((lat.reshape(-1, 1),
                            lon.reshape(-1, 1),
                            r.reshape(-1, 1)))

B_pred = secs.predict_B(pred_lat_lon_r)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
vmin, vmax = -10, 10
cmap = plt.get_cmap('RdBu_r')
t = 10

ax1.pcolormesh(obs_lon, obs_lat, B_obs[t, :, 0].reshape(obs_lon.shape),
               vmin=vmin, vmax=vmax, cmap=cmap, shading='auto')
ax2.pcolormesh(obs_lon, obs_lat, B_obs[t, :, 1].reshape(obs_lon.shape),
               vmin=vmin, vmax=vmax, cmap=cmap, shading='auto')
ax1.set_ylabel("Observations")
ax1.set_title("B$_{X}$")
ax2.set_title("B$_{Y}$")

ax1.quiver(obs_lat_lon_r[:, 1], obs_lat_lon_r[:, 0],
           B_obs[t, :, 1], B_obs[t, :, 0],
           angles='xy', scale_units='xy', scale=2)
ax2.quiver(obs_lat_lon_r[:, 1], obs_lat_lon_r[:, 0],
           B_obs[t, :, 1], B_obs[t, :, 0],
           angles='xy', scale_units='xy', scale=2)

lon = lon[..., 0]
lat = lat[..., 0]
ax3.pcolormesh(lon, lat, B_pred[t, :, 0].reshape(lon.shape),
               vmin=vmin, vmax=vmax, cmap=cmap, shading='auto')
ax4.pcolormesh(lon, lat, B_pred[t, :, 1].reshape(lon.shape),
               vmin=vmin, vmax=vmax, cmap=cmap, shading='auto')

ax3.quiver(pred_lat_lon_r[:, 1], pred_lat_lon_r[:, 0],
           B_pred[t, :, 1], B_pred[t, :, 0],
           angles='xy', scale_units='xy', scale=2)
ax4.quiver(pred_lat_lon_r[:, 1], pred_lat_lon_r[:, 0],
           B_pred[t, :, 1], B_pred[t, :, 0],
           angles='xy', scale_units='xy', scale=2)
ax3.set_ylabel("Predictions")
ax3.set_title("B$_{X}$")
ax4.set_title("B$_{Y}$")

plt.show()
