#!/usr/bin/env python3
"""
Same diagnostic pipeline as plot_transformation_steps.py -- altitude,
raw decay rate, and drag-normalized decay rate for a handful of objects --
but the last panel replaces the per-object standardized decay curves with
the consensus decay rate: every object's standardized decay-rate
measurement pooled across the population and combined with a causal,
recency-weighted average, the same method common/calculate_average_decay_original.py
uses for its population-level average (weight = exp(-gap / tau) by default).
"""
import os
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from plot_transformation_steps import dens_expo, read_sw_nrlmsise00, get_sw_params

plt.rcParams['font.family'] = 'Arial'


def main():
    get_data_from_tles_with_consensus()


def causal_weighted_average(records, eval_times, method='exponential',
                             tau_minutes=60.0, min_gap_minutes=1.0,
                             max_lookback_minutes=None):
    """Causal, recency-weighted average of pooled (t, z) measurements, evaluated
    at `eval_times`. Same weighting scheme as
    common/calculate_average_decay_original.py's causal_weighted_average,
    generalized to take an explicit evaluation grid instead of a hardcoded
    year range.
    """
    records = sorted(records, key=lambda r: r['t'])
    if not records:
        return np.full(len(eval_times), np.nan), np.zeros(len(eval_times), dtype=int)

    if max_lookback_minutes is None:
        max_lookback_minutes = 20.0 * tau_minutes if method == 'exponential' else 30.0 * 1440.0

    t0 = records[0]['t']
    t_minutes = np.array([(r['t'] - t0).total_seconds() for r in records]) / 60.0
    z_arr = np.array([r['z'] for r in records])
    n = len(t_minutes)

    eval_minutes = np.array([(t - t0).total_seconds() for t in eval_times]) / 60.0
    n_steps = len(eval_times)

    weighted_avg = np.full(n_steps, np.nan)
    counts = np.zeros(n_steps, dtype=int)

    # Sliding window over the time-sorted pooled measurements: right advances
    # to include everything <= T, left advances to drop anything older than
    # the lookback window. Same two-pointer sweep as the original function.
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

    return weighted_avg, counts


def get_data_from_tles_with_consensus():
    """
    Same data loading / preprocessing as plot_transformation_steps.get_data_from_tles(),
    plus the population-wide consensus decay rate plotted on the last panel in
    place of the individual per-object standardized decay curves.
    """
    norm_alt = True

    # ----------------------------------------------------------
    # Load interpolated altitude histories and satellite catalog
    # ----------------------------------------------------------
    file = 'data/example_objs_interp.pkl'
    with open(file, 'rb') as f:
        t, alt_by_obj, satcat = pkl.load(f)

    # ----------------------------------------------------------
    # Load or cache space weather data
    # ----------------------------------------------------------
    start_date = min(t)
    end_date = max(t)
    sw_cache = (
        'data/sw_data_'
        + start_date.strftime('%d_%m_%Y')
        + '_'
        + end_date.strftime('%d_%m_%Y')
        + '.pkl'
    )

    if os.path.exists(sw_cache):
        with open(sw_cache, 'rb') as f:
            f107A, f107, Ap, aph, t_sw = pkl.load(f)
    else:
        print('loading sw data...')
        sw_data = read_sw_nrlmsise00('data/SW-All.csv')
        f107A, f107, Ap, aph = get_sw_params(t, sw_data, 0, 0)
        with open(sw_cache, 'wb') as f:
            pkl.dump([f107A, f107, Ap, aph, t], f)

    print('loaded all data!')

    # ----------------------------------------------------------
    # Define training interval
    # ----------------------------------------------------------
    tdelta_days = 365
    start_date = dt.datetime(2023, 11, 1)

    t_ts = np.array([dt.datetime.timestamp(tt) for tt in t])
    start_idx = np.argmin(np.abs(t_ts - dt.datetime.timestamp(start_date)))

    idx_obj = np.arange(len(satcat))
    train_days = 365

    # ----------------------------------------------------------
    # Training data extraction
    # ----------------------------------------------------------
    alt_by_obj_train = alt_by_obj[start_idx:start_idx + 24 * train_days, idx_obj]
    t_train = t[start_idx:start_idx + 24 * train_days - 1]

    # ----------------------------------------------------------
    # Reference density model (exponential atmosphere)
    # ----------------------------------------------------------
    if norm_alt:
        d_ref = np.zeros((len(idx_obj), len(alt_by_obj_train) - 1))
        for i in range(len(idx_obj)):
            d_ref[i, :] = dens_expo(alt_by_obj_train[:-1, i]) * 1e9
    else:
        d_ref = np.ones((len(idx_obj), len(alt_by_obj_train) - 1))

    def compute_v_for_alt(alt):
        """Circular orbital velocity for given altitude [km]."""
        r_e = 6378.15
        mu = 398600.4418
        return np.sqrt(mu / (alt + r_e))

    # ----------------------------------------------------------
    # Drag-normalized decay computation
    # ----------------------------------------------------------
    a = alt_by_obj_train[:-1, :] + 6378.15
    mu = 398600.4418

    norm_factor_train = d_ref.T * np.sqrt(a * mu)
    dadt = np.diff(alt_by_obj_train, axis=0) / 3600
    dalt_train_raw = dadt / norm_factor_train

    # Remove positive (non-decaying) values
    dalt_train_raw_stats = np.where(dalt_train_raw > 0, np.nan, dalt_train_raw)

    dalt_mean = np.nanmean(dalt_train_raw_stats, axis=0)
    dalt_std = np.nanstd(dalt_train_raw_stats, axis=0)

    dalt_train = (dalt_train_raw_stats - dalt_mean) / dalt_std

    # ----------------------------------------------------------
    # Consensus decay rate: pool every object's standardized decay-rate
    # measurement across the whole population, then combine with a causal,
    # recency-weighted average -- same method as
    # common/calculate_average_decay_original.py's population-level average.
    # ----------------------------------------------------------
    eval_times = t_train[:-1]
    records = []
    for j in range(dalt_train.shape[1]):
        col = dalt_train[:, j]
        for k in np.flatnonzero(~np.isnan(col)):
            records.append({'t': eval_times[k], 'z': col[k]})

    consensus_decay_rate, consensus_counts = causal_weighted_average(
        records, eval_times, method='exponential', tau_minutes=60.0
    )

    # ----------------------------------------------------------
    # Diagnostic plots
    # ----------------------------------------------------------
    plt.figure(figsize=(8, 6))

    xticks = [dt.datetime(2024, 1, 1), dt.datetime(2024, 7, 1), dt.datetime(2025, 1, 1)]
    xticklabels = [f"{d.year}/{d.month}" for d in xticks]

    ax1 = plt.subplot(2, 2, 1)
    plt.plot(t_train, alt_by_obj_train[:, :10], color="#3099ac", alpha=0.3)
    plt.ylabel('Altitude [km]')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels)

    ax2 = plt.subplot(2, 2, 2)
    plt.plot(t_train[:-1], dadt[:, :10], color="#3099ac", alpha=0.3)
    plt.ylabel(r'$\dot{a}$ [km/s]')
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)

    ax3 = plt.subplot(2, 2, 3)
    plt.plot(t_train[:-1], dalt_train_raw[:, :10], color="#3099ac", alpha=0.3)
    plt.ylabel(r'$d_s$')
    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xticklabels)

    ax4 = plt.subplot(2, 2, 4)
    plt.plot(eval_times, dalt_train, color="#3099ac", alpha=0.3, zorder=1)
    plt.plot(eval_times, consensus_decay_rate, color="#00333c", linewidth=1.2,
              zorder=2, label='Weighted Average\n(Consensus Decay Rate)')
    plt.ylabel(r'$d_{sn}$')
    # A couple of objects have isolated standardized-decay spikes past -25 that
    # would otherwise flatten this whole panel; clip the view (data is untouched)
    # so the consensus line stays legible against the bulk of the population.
    ax4.set_ylim(-26, 3)
    ax4.legend(fontsize=10, loc='lower left')
    ax4.set_xticks(xticks)
    ax4.set_xticklabels(xticklabels)

    plt.tight_layout()
    plt.savefig("plots/plot_transformation_steps_consensus.png", dpi=300)
    plt.close()

    return t_train, dalt_train, consensus_decay_rate, consensus_counts


if __name__ == '__main__':
    main()
