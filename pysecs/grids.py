"""Automatic SECS grid generation from an observation network."""

import numpy as np

from pysecs.secs import calc_angular_distance


__all__ = ["make_grid", "make_image_grid"]


def make_grid(
    obs_loc: np.ndarray,
    r_shell: float,
    spacing: float | tuple[float, float] | None = None,
    padding: float | None = None,
    min_distance: float | None = None,
) -> np.ndarray:
    """Build a regular SECS grid covering an observation network.

    A grid that is too small clips currents at its edges and biases the
    fit everywhere, and a grid coarser than the station spacing cannot
    resolve the structure the stations actually see. This lays out a
    regular latitude/longitude grid with a station-informed spacing and
    padding beyond the station footprint, following the standard SECS
    grid design (e.g. Amm & Viljanen 1999; Vanhamaki & Juusola 2020).

    Parameters
    ----------
    obs_loc : ndarray (nobs, 3 [lat, lon, r])
        The observation locations the grid should cover.

    r_shell : float
        The radius of the SEC shell (e.g. R_Earth + ionospheric altitude
        for an external/ionospheric shell, or a radius below R_Earth for
        an internal/induced image shell). See :func:`make_image_grid` to
        build a matching shell at a different radius.

    spacing : float or (float, float), optional
        The grid spacing in degrees. A single float is treated as an
        angular spacing and the longitude step is scaled by
        ``1 / cos(mean observation latitude)`` so grid cells stay
        roughly square in physical distance. A ``(lat_spacing,
        lon_spacing)`` tuple sets both steps directly with no scaling.
        Default: the median nearest-neighbor separation between the
        observation locations, which is a reasonable minimum -- finer
        grids are generally safe since regularization in ``fit()``
        controls the effective degrees of freedom, but grids much
        coarser than the station spacing cannot resolve what the
        stations see. Requires at least 2 observation locations.

    padding : float, optional
        Extra margin in degrees added around the observation bounding
        box before gridding (longitude padding is scaled the same way
        as the longitude spacing). A grid clipped tightly to the
        station footprint aliases currents outside the data region onto
        its boundary poles and biases the whole fit; padding several
        grid cells beyond the stations avoids this.
        Default: 3 * lat spacing.

    min_distance : float, optional
        Any grid node within this angular distance (degrees) of an
        observation location is nudged away by shifting its latitude.
        A SEC pole is not actually singular for ground observations of
        a divergence-free system at a shell above the ground, but it is
        singular when the SEC and observation are on the same shell, and
        for a curl-free system observed from above the shell (e.g.
        satellites). This nudge is cheap insurance against those cases
        and against evaluating currents exactly at a grid node.
        Default: 1% of the (effective) latitude spacing. Pass 0 to
        disable.

    Returns
    -------
    ndarray (n, 3 [lat, lon, r])
        The generated grid locations, all at radius ``r_shell``.
    """
    obs_loc = np.atleast_2d(obs_loc)
    if obs_loc.shape[-1] != 3:
        raise ValueError("obs_loc must have 3 columns (lat, lon, r)")

    lat = obs_loc[:, 0]
    lon = obs_loc[:, 1]
    # Guard the pole where cos(lat) -> 0 would blow up the longitude spacing
    cos_lat = max(np.cos(np.deg2rad(np.mean(lat))), 1e-3)

    if spacing is None:
        if len(obs_loc) < 2:
            raise ValueError(
                "spacing must be given explicitly with fewer than 2 "
                "observation locations"
            )
        theta_deg = np.rad2deg(calc_angular_distance(obs_loc[:, :2], obs_loc[:, :2]))
        np.fill_diagonal(theta_deg, np.inf)
        lat_spacing = float(np.median(theta_deg.min(axis=1)))
        lon_spacing = lat_spacing / cos_lat
    elif isinstance(spacing, (tuple, list)):
        lat_spacing, lon_spacing = spacing
    else:
        lat_spacing = float(spacing)
        lon_spacing = lat_spacing / cos_lat

    if lat_spacing <= 0 or lon_spacing <= 0:
        raise ValueError("spacing must be positive")

    if padding is None:
        padding = 3 * lat_spacing
    lon_padding = padding / cos_lat

    lat_min = max(lat.min() - padding, -90.0)
    lat_max = min(lat.max() + padding, 90.0)
    lon_min = lon.min() - lon_padding
    lon_max = lon.max() + lon_padding

    # Anchored at the minimum bound and stepped by the exact requested
    # spacing (rather than np.linspace, which would silently stretch the
    # spacing to fit evenly between the bounds).
    n_lat = max(int(np.ceil((lat_max - lat_min) / lat_spacing)), 1) + 1
    n_lon = max(int(np.ceil((lon_max - lon_min) / lon_spacing)), 1) + 1
    lat_nodes = np.clip(lat_min + np.arange(n_lat) * lat_spacing, -90.0, 90.0)
    lon_nodes = lon_min + np.arange(n_lon) * lon_spacing

    lat_grid, lon_grid = np.meshgrid(lat_nodes, lon_nodes, indexing="ij")
    grid = np.column_stack(
        [lat_grid.ravel(), lon_grid.ravel(), np.full(lat_grid.size, float(r_shell))]
    )

    if min_distance is None:
        min_distance = 0.01 * lat_spacing
    if min_distance > 0:
        dist_deg = np.rad2deg(calc_angular_distance(grid[:, :2], obs_loc[:, :2]))
        too_close = dist_deg.min(axis=1) < min_distance
        grid[too_close, 0] += min_distance
        grid[:, 0] = np.clip(grid[:, 0], -90.0, 90.0)

    return grid


def make_image_grid(grid: np.ndarray, r_shell: float) -> np.ndarray:
    """Copy a grid's latitude/longitude nodes onto a shell at another radius.

    Useful for building a second shell at the same horizontal locations
    as an existing grid, e.g. an internal image shell below ground to
    separate induced telluric currents from the external ionospheric
    ones (Amm & Viljanen 1999).

    Parameters
    ----------
    grid : ndarray (n, 3 [lat, lon, r])
        An existing SECS grid, e.g. from :func:`make_grid`.

    r_shell : float
        The radius of the new shell.

    Returns
    -------
    ndarray (n, 3 [lat, lon, r])
        The same latitude/longitude nodes at radius ``r_shell``.
    """
    image = np.array(grid, copy=True)
    image[:, 2] = r_shell
    return image
