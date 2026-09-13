#!/usr/bin/env python3
"""
Compute a recency-weighted average of the standardized, normalized orbital
decay rate for the debris/rocket-body objects in deb_tles_2024, and plot it
across 2024.

For each object:
  1. Parse TLEs and convert mean motion to altitude (Kepler's third law).
  2. For each consecutive pair of TLEs, compute the raw altitude decay rate
     (km/s) and normalize it by an atmospheric-density/orbital-velocity
     factor (same convention as plot_transformation_steps.py's norm_factor).
  3. Discard non-decaying (positive) intervals -- e.g. station-keeping burns
     or maneuvers -- from both the per-object statistics and the final
     series, matching plot_transformation_steps.py's dalt_train_raw_stats
     masking.
  4. Standardize the remaining (decay-only) normalized decay rate to zero
     mean / unit std using that object's own 2024 statistics.

All objects' standardized, normalized decay-rate measurements are then
pooled into a single series across 2024. A per-minute evaluation grid spans
the year; at each evaluation time T, the weighted average is computed
causally over every measurement at or before T, with
    weight = 1 / max(T - measurement_timestamp, min_gap)
so a measurement from just before T counts far more than one from weeks
before it, and measurements after T are not used at all. All recency
parameters (evaluation step, min gap, exponential tau) are specified in
minutes.
"""
import os
import csv
import glob
import math
import argparse
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from find_debris_tles import parse_tle_stream, parse_fields_from_tle, mean_motion_to_altitude_km

MU = 398600.4418  # km^3/s^2
R_EARTH = 6378.15  # km

# Exponential atmosphere model (same table as traj_predict_tle.py / plot_transformation_steps.py's dens_expo)
_DENS_PARAMS = [
    (0, 25, 0, 1.225, 7.249),
    (25, 30, 25, 3.899e-2, 6.349),
    (30, 40, 30, 1.774e-2, 6.682),
    (40, 50, 40, 3.972e-3, 7.554),
    (50, 60, 50, 1.057e-3, 8.382),
    (60, 70, 60, 3.206e-4, 7.714),
    (70, 80, 70, 8.77e-5, 6.549),
    (80, 90, 80, 1.905e-5, 5.799),
    (90, 100, 90, 3.396e-6, 5.382),
    (100, 110, 100, 5.297e-7, 5.877),
    (110, 120, 110, 9.661e-8, 7.263),
    (120, 130, 120, 2.438e-8, 9.473),
    (130, 140, 130, 8.484e-9, 12.636),
    (140, 150, 140, 3.845e-9, 16.149),
    (150, 180, 150, 2.070e-9, 22.523),
    (180, 200, 180, 5.464e-10, 29.74),
    (200, 250, 200, 2.789e-10, 37.105),
    (250, 300, 250, 7.248e-11, 45.546),
    (300, 350, 300, 2.418e-11, 53.628),
    (350, 400, 350, 9.518e-12, 53.298),
    (400, 450, 400, 3.725e-12, 58.515),
    (450, 500, 450, 1.585e-12, 60.828),
    (500, 600, 500, 6.967e-13, 63.822),
    (600, 700, 600, 1.454e-13, 71.835),
    (700, 800, 700, 3.614e-14, 88.667),
    (800, 900, 800, 1.17e-14, 124.64),
    (900, 1000, 900, 5.245e-15, 181.05),
    (1000, float('inf'), 1000, 3.019e-15, 268),
]


def dens_expo(h_km):
    """Exponential-atmosphere density (kg/m^3) at altitude h_km."""
    for h_min, h_max, h0, rho0, scale_h in _DENS_PARAMS:
        if h_min <= h_km < h_max:
            return rho0 * math.exp(-(h_km - h0) / scale_h)
    return 0.0


def load_satcat_types(satcat_path='satcat.csv'):
    """Load a {norad: OBJECT_TYPE} mapping from a satcat CSV export (e.g. 'DEB', 'R/B', 'PAY')."""
    types = {}
    if not satcat_path or not os.path.exists(satcat_path):
        return types
    with open(satcat_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            norad = (row.get('NORAD_CAT_ID') or '').strip()
            if not norad:
                continue
            if norad.isdigit():
                norad = str(int(norad))
            types[norad] = (row.get('OBJECT_TYPE') or '').strip()
    return types


def load_object_series(path):
    """Return sorted, de-duplicated list of (epoch, altitude_km) for one NORAD TLE file."""
    pts = []
    with open(path, 'r') as f:
        for name, l1, l2 in parse_tle_stream(f):
            norad, ecc, mm, epoch = parse_fields_from_tle(l1, l2)
            if epoch is None or mm is None:
                continue
            alt = mean_motion_to_altitude_km(mm)
            if alt is None:
                continue
            pts.append((epoch, alt))
    pts.sort(key=lambda p: p[0])

    dedup = []
    for epoch, alt in pts:
        if dedup and dedup[-1][0] == epoch:
            continue
        dedup.append((epoch, alt))
    return dedup


def compute_normalized_decay_rate(pts, min_interval_minutes=60.0):
    """From sorted (epoch, alt) points, compute per-interval normalized decay rate.

    Returns a list of dicts: {'t': measurement_time, 'norm_rate': normalized decay rate}.

    min_interval_minutes rejects intervals shorter than this: two TLE epochs
    can be published only seconds apart (e.g. a re-issued/duplicate epoch that
    survives load_object_series's exact-timestamp dedup because the epochs
    differ by a second), and dividing a noise-level altitude difference by a
    near-zero time delta produces an enormous, physically meaningless rate
    (observed: NORAD 8756 had epochs one second apart on 2025-01-24, giving a
    computed rate of ~-285 km/day that briefly dominated the recency-weighted
    population average via a z-score of -44). Real decay-rate measurements
    need a baseline long enough for the altitude change to exceed TLE fitting
    noise; an hour is a conservative floor well below normal TLE cadence
    (typically many hours to a day) so this only screens out degenerate pairs.
    """
    records = []
    min_interval_days = min_interval_minutes / 1440.0
    for (t0, a0), (t1, a1) in zip(pts[:-1], pts[1:]):
        dt_days = (t1 - t0).total_seconds() / 86400.0
        if dt_days < min_interval_days:
            continue
        dt_s = dt_days * 86400.0
        raw_rate = (a1 - a0) / dt_s  # km/s

        rho = dens_expo(a0)
        norm_factor = rho * 1e9 * math.sqrt(MU * (a0 + R_EARTH))
        if not norm_factor:
            continue

        records.append({'t': t1, 'norm_rate': raw_rate / norm_factor})
    return records


def standardize(records):
    """Z-score normalized decay rate using this object's own mean/std.

    Non-decaying (positive) intervals -- station-keeping burns, maneuvers,
    noise -- are excluded from both the mean/std calculation and the
    returned series, matching plot_transformation_steps.py's
    dalt_train_raw_stats masking (np.where(dalt_train_raw > 0, np.nan, ...)).
    """
    rates = np.array([r['norm_rate'] for r in records])
    decay_only = np.where(rates > 0, np.nan, rates)
    mean = np.nanmean(decay_only)
    std = np.nanstd(decay_only)
    if not std or np.isnan(std):
        return None

    out = []
    for r, masked in zip(records, decay_only):
        if np.isnan(masked):
            continue
        r['z'] = (masked - mean) / std
        out.append(r)
    return out if out else None


def causal_weighted_average(all_records, step_minutes=1.0, method='inverse',
                             tau_minutes=144.0, min_gap_minutes=6.0, max_lookback_minutes=None):
    """Evaluate a causal, recency-weighted average of the standardized decay
    rate on a per-minute grid spanning `year`.

    At each evaluation time T, every measurement timestamped at or before T
    and within `max_lookback_minutes` of it is included, weighted by one of:

      method='exponential': weight = exp(-gap / tau_minutes)
          Smoothly decays with a half-life of about 0.69 * tau_minutes; larger
          tau_minutes means less dependence on recency. No floor needed since
          the weight is always finite (1 at gap=0).

      method='inverse' (default): weight = 1 / max(gap, min_gap)
          The original scheme: weight blows up sharply for very recent
          measurements and falls off slowly (harmonically) for old ones.
          min_gap floors the weight so a measurement landing (near-)exactly
          at T doesn't blow up to infinite weight.

    gap, tau_minutes, and min_gap_minutes are all in minutes.

    max_lookback_minutes bounds how far back each evaluation step looks,
    via a sliding window over the (time-sorted) measurements rather than
    re-summing every measurement since the start of the year at every step.
    Measurements older than the window contribute a weight low enough to be
    negligible for recency weighting anyway. Defaults to 20 * tau_minutes
    for the exponential method (weight < 1e-8 beyond that) and 30 days for
    the inverse method; pass a larger value (or a very large number) to
    widen it, at the cost of speed.
    """
    # Sort all pooled measurements by time once; convert timestamps to minutes
    # since the first measurement for fast vectorized arithmetic.
    all_records = sorted(all_records, key=lambda r: r['t'])
    if not all_records:
        return [], np.array([]), np.array([], dtype=int)

    if max_lookback_minutes is None:
        max_lookback_minutes = 20.0 * tau_minutes if method == 'exponential' else 30.0 * 1440.0

    t0 = all_records[0]['t']
    t_minutes = np.array([(r['t'] - t0).total_seconds() for r in all_records]) / 60.0
    z_arr = np.array([r['z'] for r in all_records])
    n = len(t_minutes)

    start_date = dt.datetime(2022, 1, 1)
    end_date = dt.datetime(2026, 1, 1)
    step = dt.timedelta(minutes=step_minutes)

    n_steps = int((end_date - start_date) / step)
    eval_times = [start_date + step * i for i in range(n_steps)]
    eval_minutes = np.array([(t - t0).total_seconds() for t in eval_times]) / 60.0

    weighted_avg = np.full(n_steps, np.nan)
    counts = np.zeros(n_steps, dtype=int)

    # Sliding window [left, right) over the time-sorted measurements: right
    # advances to include everything <= T, left advances to drop anything
    # older than the lookback window. Both pointers only move forward, so
    # the whole sweep across all evaluation steps is O(n_steps + n) instead
    # of re-scanning from the start of the year every step.
    left = 0
    right = 0
    for i, T_minutes in enumerate(eval_minutes):
        while right < n and t_minutes[right] <= T_minutes:
            right += 1
        cutoff = T_minutes - max_lookback_minutes
        while left < right and t_minutes[left] < cutoff:
            left += 1
        if right <= left:
            continue

        gaps = T_minutes - t_minutes[left:right]
        if method == 'exponential':
            weights = np.exp(-gaps / tau_minutes)
        elif method == 'inverse':
            weights = 1.0 / np.maximum(gaps, min_gap_minutes)
        else:
            raise ValueError(f"Unknown method '{method}', expected 'exponential' or 'inverse'")
        weight_sum = weights.sum()
        if weight_sum <= 0:
            continue
        weighted_avg[i] = np.dot(weights, z_arr[left:right]) / weight_sum
        counts[i] = right - left

    return eval_times, weighted_avg, counts


def main():
    p = argparse.ArgumentParser(
        description='Recency-weighted average of standardized, normalized decay rate across 2024.'
    )
    p.add_argument('--indir', default='deb_tles_all', help='Directory of per-NORAD TLE files')
    p.add_argument('--step-minutes', type=float, default=1.0,
                    help='Evaluation grid spacing in minutes (default 1, i.e. one value per minute)')
    p.add_argument('--weight-method', choices=['exponential', 'inverse'], default='exponential',
                    help="Recency weighting scheme (default 'exponential', less sensitive to recency "
                         "than the original 'inverse' 1/gap scheme)")
    p.add_argument('--tau-minutes', type=float, default=60.0,
                    help="Decay time-constant in minutes for --weight-method exponential (default 144, "
                         "i.e. 2.4 hours; larger = less dependence on recency)")
    p.add_argument('--min-gap-minutes', type=float, default=1.0,
                    help='Floor on (evaluation time - measurement time) in minutes, for --weight-method inverse (default 6)')
    p.add_argument('--max-lookback-minutes', type=float, default=None,
                    help='How far back each evaluation step looks for measurements (minutes). '
                         'Default: 20*tau for --weight-method exponential, 30 days for inverse. '
                         'Raise this if you need contributions from far-older measurements; older '
                         'measurements have negligible weight anyway for recency weighting.')
    p.add_argument('--object-type', nargs='+', default=None,
                    help="Restrict to satcat OBJECT_TYPE code(s), e.g. --object-type DEB, or "
                         "--object-type DEB R/B. Default: no filtering (use every object in --indir).")
    p.add_argument('--satcat', default='satcat.csv', help='Path to satcat CSV, used for --object-type filtering')
    p.add_argument('--out', default='plots/average_decay_final.png', help='Output plot path')
    p.add_argument('--csv-out', default=None,
                    help='Output CSV path for the evaluated series (timestamp, weighted_avg_decay_rate, '
                         'n_measurements). Default: derived from --out.')
    p.add_argument('--zoom-day', default='2024-05-11',
                    help='Also produce a second plot zoomed into this single day (YYYY-MM-DD). '
                         'Pass "" to skip.')
    p.add_argument('--zoom-out', default=None,
                    help='Output path for the zoomed single-day plot (default: derived from --out)')
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.indir, '*.txt')))
    if not paths:
        print(f"No TLE files found in {args.indir}")
        return

    if args.object_type:
        wanted = {t.upper() for t in args.object_type}
        satcat_types = load_satcat_types(args.satcat)
        before = len(paths)
        paths = [
            p for p in paths
            if satcat_types.get(os.path.splitext(os.path.basename(p))[0], '').upper() in wanted
        ]
        print(f"Filtered to OBJECT_TYPE {sorted(wanted)}: {len(paths)}/{before} objects kept.")
        if not paths:
            print("No objects match the requested --object-type filter.")
            return

    all_records = []
    per_object_records = []
    for path in paths:
        pts = load_object_series(path)
        if len(pts) < 3:
            continue
        recs = compute_normalized_decay_rate(pts)
        if not recs:
            continue
        recs = standardize(recs)
        if recs:
            all_records.extend(recs)
            per_object_records.append(recs)

    if not all_records:
        print("No decay-rate measurements computed.")
        return

    eval_times, weighted_avg, counts = causal_weighted_average(
        all_records, step_minutes=args.step_minutes, method=args.weight_method,
        tau_minutes=args.tau_minutes, min_gap_minutes=args.min_gap_minutes,
        max_lookback_minutes=args.max_lookback_minutes
    )

    if args.weight_method == 'exponential':
        weight_desc = f'exponential, tau={args.tau_minutes:.1f}min'
    else:
        weight_desc = f'inverse gap, floor={args.min_gap_minutes:.1f}min'

    fig, ax = plt.subplots(figsize=(12, 5))
    for recs in per_object_records:
        ax.plot([r['t'] for r in recs], [r['z'] for r in recs],
                color='gray', alpha=0.08, linewidth=0.4, zorder=1)
    ax.plot(eval_times, weighted_avg, color='green', linewidth=0.8, zorder=2)
    # ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Scaled, Normalized Decay Rate')
    ax.set_title(f'Recency-Weighted Average Decay Rate Across {len(paths)} Objects '
                 f'({weight_desc})')
    ax.legend(
        handles=[
            Line2D([0], [0], color='green', linewidth=0.8, label='Recency-weighted average'),
            Line2D([0], [0], color='gray', linewidth=1.2, alpha=0.6, label='Individual object decay rates'),
        ],
        loc='lower right', fontsize=8,
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    fig.tight_layout()

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    fig.savefig(args.out, dpi=150)

    csv_out = args.csv_out
    if not csv_out:
        base, _ext = os.path.splitext(args.out)
        csv_out = f"{base}.csv"
    csv_outdir = os.path.dirname(csv_out)
    if csv_outdir:
        os.makedirs(csv_outdir, exist_ok=True)
    with open(csv_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'weighted_avg_decay_rate', 'n_measurements'])
        for t, avg, cnt in zip(eval_times, weighted_avg, counts):
            writer.writerow([t.isoformat(), '' if np.isnan(avg) else avg, cnt])

    n_steps_with_data = int((counts > 0).sum())
    print(f"Saved plot to {args.out}")
    print(f"Saved series CSV to {csv_out}")
    print(f"Used {len(all_records)} decay-rate measurements from {len(paths)} objects, "
          f"{n_steps_with_data}/{len(eval_times)} evaluation steps "
          f"({args.step_minutes:.1f}-minute spacing) had data.")

    if args.zoom_day:
        zoom_start = dt.datetime.strptime(args.zoom_day, '%Y-%m-%d')
        zoom_end = zoom_start + dt.timedelta(days=1)
        eval_times_arr = np.array(eval_times)
        mask = (eval_times_arr >= zoom_start) & (eval_times_arr < zoom_end)

        if not mask.any():
            print(f"No evaluation steps found on {args.zoom_day}; skipping zoom plot.")
        else:
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            for recs in per_object_records:
                zoom_recs = [r for r in recs if zoom_start <= r['t'] < zoom_end]
                if not zoom_recs:
                    continue
                # Points only (no connecting line): consecutive records for one object
                # can be many hours apart even within this single day, and a line between
                # them would draw a long spurious diagonal across the whole panel.
                ax2.plot([r['t'] for r in zoom_recs], [r['z'] for r in zoom_recs],
                         color='gray', alpha=0.2, linewidth=0, marker='o', markersize=2.5, zorder=1)
            ax2.plot(eval_times_arr[mask], weighted_avg[mask], color='green',
                     marker='o', markersize=3, linewidth=1.0, zorder=2)
            # ax2.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax2.set_xlabel(args.zoom_day)
            ax2.set_ylabel('Scaled, Normalized Decay Rate')
            ax2.set_title(f'Causal recency-weighted average decay rate on {args.zoom_day} '
                          f'({len(paths)} objects, {args.indir}, {weight_desc})')
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig2.autofmt_xdate()
            fig2.tight_layout()

            zoom_out = args.zoom_out
            if not zoom_out:
                base, ext = os.path.splitext(args.out)
                zoom_out = f"{base}_{args.zoom_day}{ext}"
            zoom_outdir = os.path.dirname(zoom_out)
            if zoom_outdir:
                os.makedirs(zoom_outdir, exist_ok=True)
            fig2.savefig(zoom_out, dpi=150)
            print(f"Saved zoomed single-day plot to {zoom_out}")


if __name__ == '__main__':
    main()
