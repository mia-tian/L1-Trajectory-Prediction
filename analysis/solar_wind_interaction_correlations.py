"""Do RELATIONSHIPS between solar wind channels correlate with decay better than
any single channel does?

solar_wind_correlations.py answers "how well does channel X alone, at lag L,
correlate with decay" -- one line per channel, best case ~|r|=0.5-0.55 around
lag 6-9h. This script tests two ways of combining channels instead of using
them one at a time, over the same lag grid, so the peak correlations are
directly comparable to that existing ceiling:

  1. Pairwise products of z-scored channels (e.g. bz_south * proton_speed) --
     tests specific two-way interactions, including ones not already captured
     by the physics coupling functions (kan_lee, newell, ey_southward) already
     in the pipeline.
  2. PC1 of the full z-scored channel panel -- a single data-driven "joint
     storm index" that summarizes correlated co-movement across all channels
     at once, rather than picking one pair by hand.

This is a due-diligence check, not a modeling change: if nothing here clears
the existing single-channel ceiling, that's evidence the missing signal isn't
in simple same-lag combinations of these channels, and effort is better spent
elsewhere (cross-lag structure, geomagnetic indices, a narrower target, etc).
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from solar_wind_correlations import (
    DEFAULT_DSCOVR_CSV,
    DEFAULT_SW_FILE,
    build_weighted_decay_proxy,
    hourly_solar_wind_series,
    load_dscovr,
    load_solar_wind,
)

DEFAULT_OUTPUT_CSV_PAIRS = "plots/solar_wind_interaction_pair_correlations.csv"
DEFAULT_OUTPUT_CSV_PC1 = "plots/solar_wind_interaction_pc1_correlations.csv"
DEFAULT_OUTPUT_PNG = "plots/solar_wind_interaction_correlations.png"

# Primitive measured channels only -- the already-derived coupling functions
# (kan_lee_mvm, newell_coupling, ey_southward_mvm, bs) are themselves specific
# physically-chosen combinations of these, so they're excluded here to avoid
# testing "a combination of a combination" and to isolate genuinely new pairs.
CANDIDATE_CHANNELS = [
    "bt", "by_gsm", "bz_gsm", "proton_speed", "proton_density",
    "proton_temperature", "dynamic_pressure_npa",
]


def lagged_corr(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    if lag == 0:
        a, b = x, y
    else:
        a, b = x[:-lag], y[lag:]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def build_channel_matrix(solar_wind: pd.DataFrame, columns: list[str], grid: pd.DatetimeIndex) -> pd.DataFrame:
    data = {}
    for col in columns:
        series = hourly_solar_wind_series(solar_wind, col, grid)  # already flips bz_gsm -> southward-positive
        if col == "bz_gsm":
            col = "bz_south"
        data[col] = series
    return pd.DataFrame(data, index=grid)


def pairwise_interaction_scan(channel_df: pd.DataFrame, decay_proxy: np.ndarray, max_lag_hours: int) -> pd.DataFrame:
    z = (channel_df - channel_df.mean()) / channel_df.std()
    results = []
    for a, b in combinations(channel_df.columns, 2):
        prod = (z[a] * z[b]).to_numpy()
        for lag in range(0, max_lag_hours + 1):
            results.append({"pair": f"{a}*{b}", "lag_hours": lag,
                             "correlation": lagged_corr(prod, decay_proxy, lag)})
    return pd.DataFrame(results)


def pc1_scan(channel_df: pd.DataFrame, decay_proxy: np.ndarray, max_lag_hours: int) -> tuple[pd.DataFrame, pd.Series]:
    z = (channel_df - channel_df.mean()) / channel_df.std()
    valid_mask = z.notna().all(axis=1)
    u, s, vt = np.linalg.svd(z.loc[valid_mask].to_numpy(), full_matrices=False)
    pc1 = pd.Series(np.nan, index=z.index)
    pc1.loc[valid_mask] = u[:, 0] * s[0]
    loadings = pd.Series(vt[0], index=channel_df.columns)
    # orient sign so PC1 rises with "more geoeffective" conditions (positive dynamic pressure loading)
    if loadings.get("dynamic_pressure_npa", 0.0) < 0:
        pc1 = -pc1
        loadings = -loadings
    pc1_arr = pc1.to_numpy()
    results = [{"lag_hours": lag, "correlation": lagged_corr(pc1_arr, decay_proxy, lag)}
               for lag in range(0, max_lag_hours + 1)]
    return pd.DataFrame(results), loadings


def plot_comparison(pair_df: pd.DataFrame, pc1_df: pd.DataFrame, output_path: Path, max_hours: int) -> None:
    best_pair_name = pair_df.loc[pair_df["correlation"].abs().idxmax(), "pair"]
    best_pair = pair_df[pair_df["pair"] == best_pair_name].sort_values("lag_hours")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(best_pair["lag_hours"], best_pair["correlation"], marker="o", markersize=3,
            label=f"best pair: {best_pair_name}", color="tab:red")
    ax.plot(pc1_df["lag_hours"], pc1_df["correlation"], marker="o", markersize=3,
            label="PC1 (joint mode)", color="tab:purple")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(0.55, color="gray", linewidth=0.8, linestyle=":", label="single-channel ceiling (~0.55)")
    ax.axhline(-0.55, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlim(0, max_hours)
    ax.set_ylim(-0.7, 0.7)
    ax.set_xticks(range(0, max_hours + 1))
    ax.set_xlabel("Lag from solar wind measurement to decay [hours]")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Best pairwise interaction & PC1 vs. single-channel ceiling")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decay-indir", default="deb_tles_2024")
    parser.add_argument("--decay-satcat", default="satcat.csv")
    parser.add_argument("--decay-object-type", nargs="+", default=None)
    parser.add_argument("--decay-weight-method", choices=["exponential", "inverse"], default="exponential")
    parser.add_argument("--decay-tau-minutes", type=float, default=60.0)
    parser.add_argument("--decay-min-gap-minutes", type=float, default=6.0)
    parser.add_argument("--decay-step-minutes", type=float, default=60.0)
    parser.add_argument("--decay-year", type=int, default=2024)
    parser.add_argument("--sw-source", choices=["omni", "dscovr"], default="dscovr")
    parser.add_argument("--sw-file", default=DEFAULT_SW_FILE)
    parser.add_argument("--dscovr-csv", default=DEFAULT_DSCOVR_CSV)
    parser.add_argument("--max-lag-hours", type=int, default=24)
    parser.add_argument("--output-png", default=DEFAULT_OUTPUT_PNG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent  # repo root (this file now lives in analysis/)

    decay_index, decay_proxy = build_weighted_decay_proxy(
        indir=root / args.decay_indir,
        satcat_path=root / args.decay_satcat,
        object_types=args.decay_object_type,
        weight_method=args.decay_weight_method,
        tau_minutes=args.decay_tau_minutes,
        min_gap_minutes=args.decay_min_gap_minutes,
        step_minutes=args.decay_step_minutes,
        year=args.decay_year,
    )

    solar_wind = load_dscovr(root / args.dscovr_csv) if args.sw_source == "dscovr" else load_solar_wind(root / args.sw_file)

    columns = [c for c in CANDIDATE_CHANNELS if c in solar_wind.columns]
    missing = [c for c in CANDIDATE_CHANNELS if c not in solar_wind.columns]
    if missing:
        print(f"Note: skipping unavailable channels: {missing}")

    channel_df = build_channel_matrix(solar_wind, columns, decay_index)

    print(f"Scanning {len(list(combinations(channel_df.columns, 2)))} pairs x {args.max_lag_hours + 1} lags...")
    pair_df = pairwise_interaction_scan(channel_df, decay_proxy, args.max_lag_hours)
    pc1_df, loadings = pc1_scan(channel_df, decay_proxy, args.max_lag_hours)

    root_plots = root / "plots"
    root_plots.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(root / DEFAULT_OUTPUT_CSV_PAIRS, index=False)
    pc1_df.to_csv(root / DEFAULT_OUTPUT_CSV_PC1, index=False)
    plot_comparison(pair_df, pc1_df, root / args.output_png, max_hours=args.max_lag_hours)

    print("\nTop 10 pairwise interactions by |peak correlation| (best lag each):")
    best_per_pair = pair_df.loc[pair_df.groupby("pair")["correlation"].apply(lambda s: s.abs().idxmax())]
    best_per_pair = best_per_pair.reindex(best_per_pair["correlation"].abs().sort_values(ascending=False).index)
    for row in best_per_pair.head(10).itertuples():
        print(f"  {row.pair:35s} lag={int(row.lag_hours):2d}h  r={row.correlation:+.3f}")

    best_pc1 = pc1_df.loc[pc1_df["correlation"].abs().idxmax()]
    print(f"\nPC1 (joint mode) best: lag={int(best_pc1['lag_hours'])}h  r={best_pc1['correlation']:+.3f}")
    print("PC1 loadings (sign-oriented, positive = more geoeffective):")
    for name, val in loadings.sort_values(key=np.abs, ascending=False).items():
        print(f"  {name:22s} {val:+.3f}")

    print(f"\nSaved plot to {args.output_png}")
    print(f"Saved pairwise scan to {DEFAULT_OUTPUT_CSV_PAIRS}")
    print(f"Saved PC1 scan to {DEFAULT_OUTPUT_CSV_PC1}")


if __name__ == "__main__":
    main()
