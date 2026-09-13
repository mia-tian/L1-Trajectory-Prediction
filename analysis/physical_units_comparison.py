"""Convert the z-score prediction-error comparison into physical units:
decay-rate error (km/day) and, via along_track_deviation_methodology.tex's
existing formula, the resulting along-track position error (km) if that
(wrong) decay-rate prediction were used to propagate the object forward over
the backtest horizon. This is the number that's actually comparable to an
operational conjunction-assessment threshold -- z-score RMSE alone can't
answer "is this good enough."

Along-track sensitivity (see along_track_deviation_methodology.tex):
  n = sqrt(mu / r^3)                     mean motion [rad/s]
  ndot = -(3/2) * n * (rdot / r)         decay-driven mean-motion rate
  s(t) = 0.5 * r * ndot * t^2            along-track displacement after t

Differentiating s(t) w.r.t. a decay-rate ERROR delta_rdot (holding r, n fixed
-- the error itself is small relative to r) gives the along-track POSITION
error due to that decay-rate error alone:
  delta_s(t) = -0.75 * n * t^2 * delta_rdot
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
for _sub in ("common", "population_proxy", "individual_decay", "data_prep", "analysis"):
    _p = str(_Path(__file__).resolve().parent.parent / _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from calculate_average_decay import (
    compute_normalized_decay_rate,
    dens_expo,
    load_object_series,
)
from infer_decay_from_proxy import causal_asof_lookup

MU = 398600.4418  # km^3/s^2
R_EARTH = 6378.15  # km

RAW_CSV = "plots/factor_model_comparison_raw.csv"
HORIZON_EDGES = [2, 4, 8, 16, 24]


def object_std_k(indir: Path, norad: str) -> float:
    """Std of decay-only norm_rate for this object (same stat standardize() uses)."""
    pts = load_object_series(str(indir / f"{norad}.txt"))
    recs = compute_normalized_decay_rate(pts)
    rates = np.array([r["norm_rate"] for r in recs])
    decay_only = np.where(rates > 0, np.nan, rates)
    return float(np.nanstd(decay_only))


def object_altitude_series(indir: Path, norad: str):
    pts = load_object_series(str(indir / f"{norad}.txt"))
    times = [p[0] for p in pts]
    alts = np.array([p[1] for p in pts])
    return times, alts


def main() -> None:
    df = pd.read_csv(RAW_CSV, parse_dates=["t0"])
    indir = Path("deb_tles_2024")

    std_k_cache: dict[str, float] = {}
    alt_series_cache: dict[str, tuple] = {}

    delta_rdot_old_kmday, delta_rdot_new_kmday = [], []
    delta_s_old_km, delta_s_new_km = [], []
    alt_used = []

    for row in df.itertuples():
        norad = str(row.norad)
        if norad not in std_k_cache:
            std_k_cache[norad] = object_std_k(indir, norad)
        if norad not in alt_series_cache:
            alt_series_cache[norad] = object_altitude_series(indir, norad)
        std_k = std_k_cache[norad]
        times, alts = alt_series_cache[norad]

        target_t = row.t0 + pd.Timedelta(hours=row.horizon_hours)
        alt = float(causal_asof_lookup(times, alts, [target_t])[0])
        if not np.isfinite(alt):
            delta_rdot_old_kmday.append(np.nan)
            delta_rdot_new_kmday.append(np.nan)
            delta_s_old_km.append(np.nan)
            delta_s_new_km.append(np.nan)
            alt_used.append(np.nan)
            continue
        alt_used.append(alt)

        r = alt + R_EARTH  # km, orbital radius (near-circular approx)
        rho = dens_expo(alt)
        norm_factor = rho * 1e9 * np.sqrt(MU * r)  # same convention as compute_normalized_decay_rate
        n = np.sqrt(MU / r ** 3)  # rad/s

        # z-score error -> normalized-decay-rate error -> physical decay-rate error (km/s)
        err_old_z = row.pred_mean - row.truth
        err_new_z = row.new_model_pred - row.truth
        delta_rdot_old = err_old_z * std_k * norm_factor  # km/s
        delta_rdot_new = err_new_z * std_k * norm_factor  # km/s
        delta_rdot_old_kmday.append(delta_rdot_old * 86400.0)
        delta_rdot_new_kmday.append(delta_rdot_new * 86400.0)

        t_s = row.horizon_hours * 3600.0
        delta_s_old = -0.75 * n * t_s ** 2 * delta_rdot_old
        delta_s_new = -0.75 * n * t_s ** 2 * delta_rdot_new
        delta_s_old_km.append(delta_s_old)
        delta_s_new_km.append(delta_s_new)

    df["altitude_km"] = alt_used
    df["decay_rate_err_old_km_per_day"] = delta_rdot_old_kmday
    df["decay_rate_err_new_km_per_day"] = delta_rdot_new_kmday
    df["along_track_err_old_km"] = delta_s_old_km
    df["along_track_err_new_km"] = delta_s_new_km
    df = df.dropna(subset=["along_track_err_old_km", "along_track_err_new_km"])

    edges = sorted(set(HORIZON_EDGES) | {0.0})
    labels = [f"{edges[i]:g}-{edges[i + 1]:g}h" for i in range(len(edges) - 1)]
    df["horizon_bucket"] = pd.cut(df["horizon_hours"], bins=edges, labels=labels, include_lowest=True)

    print(f"Median altitude in sample: {df['altitude_km'].median():.0f} km\n")

    print("Decay-rate prediction error, physical units (km/day):")
    rows = []
    for bucket, g in df.groupby("horizon_bucket", observed=True):
        rows.append({
            "horizon_bucket": bucket, "n": len(g),
            "rmse_decay_old_km_per_day": np.sqrt(np.mean(g["decay_rate_err_old_km_per_day"] ** 2)),
            "rmse_decay_new_km_per_day": np.sqrt(np.mean(g["decay_rate_err_new_km_per_day"] ** 2)),
            "rmse_alongtrack_old_km": np.sqrt(np.mean(g["along_track_err_old_km"] ** 2)),
            "rmse_alongtrack_new_km": np.sqrt(np.mean(g["along_track_err_new_km"] ** 2)),
            "median_abs_alongtrack_old_km": g["along_track_err_old_km"].abs().median(),
            "median_abs_alongtrack_new_km": g["along_track_err_new_km"].abs().median(),
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    thresh = df["truth"].abs().quantile(0.90)
    extreme = df[df["truth"].abs() >= thresh]
    print(f"\nExtreme-event slice (|truth| z >= {thresh:.2f}, n={len(extreme)}):")
    print(f"  along-track RMSE: old={np.sqrt(np.mean(extreme['along_track_err_old_km']**2)):.2f} km  "
          f"new={np.sqrt(np.mean(extreme['along_track_err_new_km']**2)):.2f} km")
    print(f"  decay-rate RMSE:  old={np.sqrt(np.mean(extreme['decay_rate_err_old_km_per_day']**2)):.3f} km/day  "
          f"new={np.sqrt(np.mean(extreme['decay_rate_err_new_km_per_day']**2)):.3f} km/day")

    df.to_csv("plots/physical_units_comparison_raw.csv", index=False)
    print(f"\nSaved detailed per-case physical-unit results to plots/physical_units_comparison_raw.csv")


if __name__ == "__main__":
    main()
