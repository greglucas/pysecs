"""Simple wall-clock benchmarks for pysecs.

Run from the repository root (no extra dependencies required):

    python benchmarks/bench_secs.py
    python benchmarks/bench_secs.py --quick

Each case reports the best of ``--repeat`` runs. Sizes are chosen to be
representative of a regional analysis: a grid of divergence-free SECs,
a few tens of ground stations, and thousands of time steps.
"""

import argparse
import time
from collections.abc import Callable

import numpy as np

import pysecs


R_EARTH = 6371e3


def build_system(nsec_side: int, nobs: int, rng: np.random.Generator) -> tuple:
    """Build SEC grid, observation locations, and prediction grid."""
    lats = np.linspace(30, 60, nsec_side)
    lons = np.linspace(-20, 30, nsec_side)
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
    sec_locs = np.column_stack(
        [lat_g.ravel(), lon_g.ravel(), np.full(lat_g.size, R_EARTH + 110e3)]
    )
    obs_locs = np.column_stack(
        [
            rng.uniform(32, 58, nobs),
            rng.uniform(-18, 28, nobs),
            np.full(nobs, R_EARTH),
        ]
    )
    plats = np.linspace(32, 58, 50)
    plons = np.linspace(-18, 28, 50)
    plat_g, plon_g = np.meshgrid(plats, plons, indexing="ij")
    pred_locs = np.column_stack(
        [plat_g.ravel(), plon_g.ravel(), np.full(plat_g.size, R_EARTH)]
    )
    return sec_locs, obs_locs, pred_locs


def best_of(func: Callable[[], object], repeat: int) -> float:
    """Return the best wall-clock time of ``repeat`` calls."""
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    """Run the benchmark suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller sizes")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    nsec_side = 15 if args.quick else 30
    nobs = 30 if args.quick else 60
    ntimes = 500 if args.quick else 2000
    npatterns = 8

    rng = np.random.default_rng(0)
    sec_locs, obs_locs, pred_locs = build_system(nsec_side, nobs, rng)
    nsec = sec_locs.shape[0]

    obs_B = rng.normal(0, 100e-9, (ntimes, nobs, 3))
    # Recurring station-dropout patterns (e.g. intermittent data gaps)
    obs_std = np.ones_like(obs_B)
    for i in range(npatterns):
        obs_std[i::npatterns, i % nobs, :] = np.inf

    print(f"nsec={nsec} nobs={nobs} ntimes={ntimes} npred={len(pred_locs)}")
    print(f"{'case':<40}{'best time':>12}")

    def bench(name: str, func: Callable[[], object]) -> None:
        print(f"{name:<40}{best_of(func, args.repeat):>10.3f} s")

    bench("T_df transfer matrix", lambda: pysecs.T_df(obs_locs, sec_locs))

    def fit_uniform() -> None:
        secs = pysecs.SECS(sec_df_loc=sec_locs)
        secs.fit(obs_locs, obs_B)

    bench("fit: uniform std (1 SVD)", fit_uniform)

    def fit_patterns() -> None:
        secs = pysecs.SECS(sec_df_loc=sec_locs)
        secs.fit(obs_locs, obs_B, obs_std=obs_std)

    bench(f"fit: {npatterns} recurring std patterns", fit_patterns)

    secs = pysecs.SECS(sec_df_loc=sec_locs)
    secs.fit(obs_locs, obs_B)
    # Warm the prediction cache outside the timed region once
    secs.predict_B(pred_locs)

    def predict_warm() -> None:
        secs.predict_B(pred_locs)

    bench("predict_B: warm transfer cache", predict_warm)

    def predict_cold() -> None:
        secs._pred_loc_B = None
        secs._T_pred_B = None
        secs.predict_B(pred_locs)

    bench("predict_B: cold transfer cache", predict_cold)


if __name__ == "__main__":
    main()
