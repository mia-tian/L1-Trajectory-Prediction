#!/usr/bin/env python3
"""Identify stormtime segments in the recency-weighted average decay-rate
series produced by calculate_average_decay_original.py.

A "stormtime segment" is a maximal contiguous run of evaluation steps where
the scaled, normalized decay rate drops below --threshold (default -2.0),
padded by --pad-hours (default 24h) on both ends to also capture the
run-up and recovery around the storm, then merged wherever that padding
causes segments to overlap.

The resulting segments are written to a CSV (start, end, duration_hours)
for use by gbm_stormtime_solar_wind_to_decay.py, and plotted against the
full decay-rate series with shaded backgrounds marking stormtime.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_series(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp")
    return pd.Series(
        df["weighted_avg_decay_rate"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(df["timestamp"]),
        name="decay_w",
    )


def find_raw_segments(series: pd.Series, threshold: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Maximal contiguous runs (by row position in the input grid) below threshold."""
    below = (series < threshold).to_numpy()
    idx = np.flatnonzero(below)
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, breaks)
    times = series.index
    return [(times[g[0]], times[g[-1]]) for g in groups]


def pad_and_merge(
    segments: list[tuple[pd.Timestamp, pd.Timestamp]], pad_hours: float
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not segments:
        return []
    pad = pd.Timedelta(hours=pad_hours)
    padded = sorted(((s - pad, e + pad) for s, e in segments), key=lambda x: x[0])
    merged = [padded[0]]
    for s, e in padded[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def write_segments_csv(path: str, segments: list[tuple[pd.Timestamp, pd.Timestamp]]) -> None:
    outdir = Path(path).parent
    if str(outdir):
        outdir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment_id", "start", "end", "duration_hours"])
        for i, (s, e) in enumerate(segments):
            writer.writerow([i, s.isoformat(), e.isoformat(), (e - s).total_seconds() / 3600.0])


# Validated palette (dataviz skill, light mode) -- kept as plain constants since
# this is a static matplotlib figure, not a themable HTML chart.
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"
_SERIES_BLUE = "#2a78d6"   # categorical slot 1 -- the decay-rate line
_STORM_GREEN = "#0ca30c"  # storm-segment shading


def plot_segments(
    series: pd.Series,
    segments: list[tuple[pd.Timestamp, pd.Timestamp]],
    threshold: float,
    pad_hours: float,
    out_path: str,
    resample: str | None,
) -> None:
    plot_series = series.resample(resample).mean() if resample else series

    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans", "sans-serif"]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    for i, (s, e) in enumerate(segments):
        ax.axvspan(s, e, color=_STORM_GREEN, alpha=0.12, linewidth=0,
                   label="High Drag Segment" if i == 0 else None, zorder=1)

    ax.axhline(0, color=_GRIDLINE, linewidth=1.0, zorder=2)
    ax.axhline(threshold, color=_INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
               label=f"Threshold ({threshold:g})", zorder=2)
    ax.plot(plot_series.index, plot_series.to_numpy(), color=_SERIES_BLUE,
             linewidth=0.7, solid_capstyle="round", label="Consensus Decay Rate", zorder=3)

    ax.set_xlabel("Time", fontsize=13, color=_INK_SECONDARY, labelpad=8)
    ax.set_ylabel("Scaled, Normalized Decay Rate", fontsize=13, color=_INK_SECONDARY, labelpad=8)
    fig.suptitle("Stormtime Segments in the Debris Decay-Rate Series", x=0.085, y=0.98,
                 ha="left", fontsize=19, fontweight="bold", color=_INK_PRIMARY)
    ax.set_title(
        f"Decay Rate < {threshold:g}, Padded ±{pad_hours:g}h → {len(segments)} Merged Segments",
        loc="left", fontsize=12, color=_INK_MUTED, pad=44,
    )

    legend = ax.legend(fontsize=12, loc="lower left", bbox_to_anchor=(0.0, 1.02),
                        ncol=3, frameon=False, labelcolor=_INK_SECONDARY,
                        handlelength=1.6, columnspacing=1.6, borderaxespad=0.0)
    for text in legend.get_texts():
        text.set_color(_INK_SECONDARY)

    ax.set_xlim(plot_series.index.min(), plot_series.index.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="both", which="both", length=0, labelsize=11, colors=_INK_MUTED)

    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_BASELINE)
    ax.spines["bottom"].set_linewidth(1.0)

    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    outdir = Path(out_path).parent
    if str(outdir):
        outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=_SURFACE)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="plots/average_decay_final.csv",
                    help="Input CSV produced by calculate_average_decay_original.py")
    p.add_argument("--threshold", type=float, default=-.2,
                    help="Stormtime threshold on the scaled, normalized decay rate (default -2.0)")
    p.add_argument("--pad-hours", type=float, default=36.0,
                    help="Hours to pad onto the start and end of each raw sub-threshold run (default 24)")
    p.add_argument("--out-csv", default="population_proxy/stormtime_segments.csv",
                    help="Output CSV of merged, padded stormtime segments")
    p.add_argument("--plot", default="plots/stormtime_segments.png", help="Output plot path")
    p.add_argument("--plot-resample", default="1h",
                    help="Resample rule for the overview plot, e.g. '1h' or '15min' "
                         "(the raw series is far too dense to plot directly over multiple years). "
                         "Pass '' to plot the raw series unresampled.")
    args = p.parse_args()

    series = load_series(args.csv)
    raw_segments = find_raw_segments(series, args.threshold)
    segments = pad_and_merge(raw_segments, args.pad_hours)

    total_hours = sum((e - s).total_seconds() / 3600.0 for s, e in segments)
    span_hours = (series.index.max() - series.index.min()).total_seconds() / 3600.0
    print(f"Found {len(raw_segments)} raw sub-threshold runs -> {len(segments)} merged, "
          f"padded (±{args.pad_hours:g}h) stormtime segments.")
    print(f"Stormtime coverage: {total_hours:.1f}h / {span_hours:.1f}h "
          f"({100.0 * total_hours / span_hours:.1f}% of the series).")

    write_segments_csv(args.out_csv, segments)
    print(f"Saved segments to {args.out_csv}")

    plot_segments(series, segments, args.threshold, args.pad_hours, args.plot,
                  resample=args.plot_resample or None)
    print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
