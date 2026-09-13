"""Stormtime-only multi-horizon LightGBM forecast of solar-wind-driven
orbital decay.

Architecture: same Stage 1 Ridge-on-decay-momentum + Stage 2 LightGBM-on-
solar-wind-residual design as gbm_single_horizon.py (same trailing-window
feature engineering, delta framing, event weighting), but Stage 2 predicts a
WHOLE WINDOW of horizons (1..--out-hours) at once with a single shared,
horizon-conditioned LightGBM model (horizon appended as a feature, same
design as gbm_solar_wind_to_decay_v2.py) instead of one fixed horizon. This
lets every diagnostic -- RMSE, R^2, delta correlation -- be reported and
plotted AS A FUNCTION OF HORIZON, not just at one arbitrarily chosen lead
time. Stage 2's contribution IS scaled by a per-horizon validation-fit gate
(fit_solar_wind_gate(), same fitting scheme as v2), motivated by delta
correlation looking reasonable at 6-10h while RMSE/R^2 still degraded there
under a single fixed scalar -- i.e. an amplitude/calibration problem, not a
directional one. v2's docstring flagged this exact gate mechanism as a
documented overfitting failure mode (pinned at its ceiling on that script's
817-feature space); watch the printed gate values and the WARNING when they
pin at --max-gate here too.

Training is restricted to hourly forecast origins inside a stormtime segment
(from find_stormtime_segments.py's output CSV: runs where the scaled,
normalized decay rate dropped below a threshold, padded +-24-36h), so the
model specializes on storm-driven decay behavior instead of being diluted by
the much larger quiet-time majority of 2022-2025.

Segment-level train/val/test split
-----------------------------------
A plain chronological cutoff applied row-by-row can slice a single storm
segment in half (part before --val-date in train, the rest of that SAME
storm after it in val/test) -- the two halves share almost all of their
recent decay/solar-wind history, so that "split" barely tests anything.
Instead, whole SEGMENTS are assigned to train/val/test by each segment's
start time vs --val-date/--split-date, and every row from that segment goes
wherever its segment went. No segment is split across two buckets, and rows
are never treated as one undifferentiated pool that ignores which storm they
came from. Non-stormtime hours are dropped entirely, in every split.

With the default --val-date/--split-date, training spans essentially all of
2022-2025 (38/43 segments, 2022-12 through 2025-06), holding out only the
most recent handful of storms for validation (3 segments, mid-late 2025) and
test (2 segments, December 2025).

Validation plots
-----------------
--plot: RMSE / R^2 / delta-correlation, each plotted BY HORIZON (1..
--out-hours), each panel comparing combined vs. Stage1 (momentum) vs.
persistence, plus (on the delta-correlation panel) each individual solar-wind
CHANNEL's own naive single-variable correlation with the actual change --
"solar wind in general" (the fitted model, bold lines) alongside "each
component" (individual channels, thin lines), the same per-parameter framing
solar_wind_correlations.py's plot_lag_lines uses. Plus feature importances.
--zoom-plot: one panel per TEST stormtime segment, zoomed to that segment's
own start/end +- margin, instead of one panel spanning the whole test
period. Each panel overlays the COMPLETE 1..--out-hours forecast trajectory
("fan") launched from many origins across the segment (--zoom-fan-stride-
hours apart), not just one arbitrarily chosen horizon.

Usage
-----
python gbm_stormtime_solar_wind_to_decay.py train
python gbm_stormtime_solar_wind_to_decay.py train --out-hours 10 --zoom-fan-stride-hours 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import lightgbm as lgb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import sys as _sys
from pathlib import Path as _Path
for _sub in ("common", "population_proxy", "individual_decay", "data_prep", "analysis"):
    _p = str(_Path(__file__).resolve().parent.parent / _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from calculate_average_decay_original import (
    causal_weighted_average,
    compute_normalized_decay_rate,
    load_object_series,
    load_satcat_types,
)
from solar_wind_correlations import prettify_engineered_feature_name, hourly_solar_wind_series

# The stormtime segments span 2022-2025, so (unlike gbm_single_horizon.py's
# 2024-only defaults) this needs the multi-year debris catalog and DSCOVR CSV.
DEFAULT_INDIR = "deb_tles_all"
DEFAULT_SATCAT = "satcat.csv"
DEFAULT_DSCOVR_CSV = "dscovr_2022_2026/dscovr_2022_2026_core_and_derived.csv"
DEFAULT_SEGMENTS_CSV = "population_proxy/stormtime_segments.csv"

OUT_HOURS = 8

SW_FEATURES = [
    "proton_speed", "proton_density", "proton_temperature",
    "bt", "by_gsm", "bz_gsm",
    "bs", "dynamic_pressure_npa", "ey_southward_mvm",
    "kan_lee_mvm", "newell_coupling", "akasofu_epsilon_gw",
    "ring_current_proxy_tau3h", "ring_current_proxy_tau8h", "ring_current_proxy_obrien",
]

# See gbm_single_horizon.py's module comment for the collinearity analysis
# behind this trim; kept here for the same --feature-set ablation switch.
TRIMMED_SW_FEATURES = [
    "proton_speed", "proton_density", "proton_temperature",
    "bt", "by_gsm", "bz_gsm",
    "dynamic_pressure_npa", "newell_coupling",
    "ring_current_proxy_tau3h", "ring_current_proxy_tau8h", "ring_current_proxy_obrien",
]

DECAY_TREND_HOURS = (1, 3, 6, 12)

QUALITY_CANDIDATES = (
    "quality_combined", "overall_quality", "overall_quality_fc",
    "overall_quality_mag", "quality_fc", "quality_mag",
)


def decay_trend_hours_for(decay_history_hours: int) -> tuple[int, ...]:
    """Same scaling as gbm_single_horizon.py's decay_trend_hours_for."""
    fractions = (1 / 12, 1 / 4, 1 / 2, 1.0)
    hours = sorted({max(1, round(decay_history_hours * f)) for f in fractions})
    return tuple(hours)


# ---------------------------------------------------------------------------
# Target: train-only-normalized decay proxy (same as gbm_single_horizon.py)
# ---------------------------------------------------------------------------

def normalize_object_records_train_only(records: list[dict], cutoff: pd.Timestamp) -> list[dict] | None:
    valid = [r for r in records if np.isfinite(r.get("norm_rate", np.nan)) and r["norm_rate"] <= 0]
    if not valid:
        return None
    train_values = np.array(
        [r["norm_rate"] for r in valid if pd.Timestamp(r["t"]) < cutoff], dtype=float,
    )
    if train_values.size < 3:
        return None
    mean = float(np.mean(train_values))
    std = float(np.std(train_values))
    if not np.isfinite(std) or std <= 0:
        return None
    output = []
    for record in valid:
        copied = dict(record)
        copied["z"] = (float(record["norm_rate"]) - mean) / std
        output.append(copied)
    return output


def build_decay_series(indir: str, satcat_path: str, object_types,
                        weight_method: str, tau_minutes: float, min_gap_minutes: float,
                        normalization_cutoff: pd.Timestamp) -> pd.Series:
    paths = sorted(glob.glob(os.path.join(indir, "*.txt")))
    if object_types:
        wanted = {t.upper() for t in object_types}
        satcat_types = load_satcat_types(satcat_path)
        paths = [p for p in paths if satcat_types.get(os.path.splitext(os.path.basename(p))[0], "").upper() in wanted]

    all_records = []
    kept_objects = 0
    for path in paths:
        pts = load_object_series(path)
        if len(pts) < 3:
            continue
        recs = compute_normalized_decay_rate(pts)
        standardized = normalize_object_records_train_only(recs, normalization_cutoff)
        if standardized:
            all_records.extend(standardized)
            kept_objects += 1
    if not all_records:
        raise RuntimeError(f"No decay-rate measurements computed from {indir}")

    eval_times, weighted_avg, _counts = causal_weighted_average(
        all_records, step_minutes=60.0,
        method=weight_method, tau_minutes=tau_minutes, min_gap_minutes=min_gap_minutes,
    )
    print(f"Decay target uses {len(all_records)} intervals from {kept_objects}/{len(paths)} objects, "
          f"normalized using records before {normalization_cutoff}")
    return pd.Series(weighted_avg, index=pd.DatetimeIndex(eval_times), name="decay_w")


# ---------------------------------------------------------------------------
# Solar wind loading (same as gbm_single_horizon.py)
# ---------------------------------------------------------------------------

def _choose_quality_column(frame: pd.DataFrame) -> str | None:
    for column in QUALITY_CANDIDATES:
        if column in frame.columns:
            return column
    return None


def load_solar_wind_minute(dscovr_csv: str, max_fill_minutes: int = 15) -> pd.DataFrame:
    frame = pd.read_csv(dscovr_csv, parse_dates=["time"])
    missing = [c for c in SW_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"DSCOVR CSV is missing required fields: {missing}")
    frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None)
    frame = frame.sort_values("time").drop_duplicates("time").set_index("time")

    science = frame[SW_FEATURES].replace([-99999, -99999.0], np.nan).copy()
    quality_column = _choose_quality_column(frame)
    if quality_column is not None:
        bad = pd.to_numeric(frame[quality_column], errors="coerce") >= 1
        science.loc[bad, :] = np.nan
        print(f"Applied DSCOVR quality mask from {quality_column!r}")

    full_index = pd.date_range(science.index.min(), science.index.max(), freq="min")
    science = science.reindex(full_index)
    science = science.ffill(limit=max_fill_minutes)
    return science


# ---------------------------------------------------------------------------
# Features: exactly a 12h (--input-hours) window, two representations
# (identical to gbm_single_horizon.py -- see that file for full rationale)
# ---------------------------------------------------------------------------

def build_features(
    sw_minute: pd.DataFrame,
    decay: pd.Series,
    input_hours: int,
    decay_history_hours: int,
    lag_features: str = "both",
    decay_trend_hours=DECAY_TREND_HOURS,
    short_window_hours: int = 0,
    trend_hours: int = 0,
    n_subwindows: int = 1,
    sw_features=None,
    include_relax_ratio: bool = True,
    include_now_feature: bool = True,
    window_stats: str = "minmax",
    shock_features: bool = True,
    shock_hours: int = 1,
) -> pd.DataFrame:
    if lag_features not in ("both", "window", "lags", "none"):
        raise ValueError(f"Unknown --lag-features {lag_features!r}")
    if window_stats not in ("minmax", "richer"):
        raise ValueError(f"Unknown --window-stats {window_stats!r}")
    if input_hours % n_subwindows != 0:
        raise ValueError(f"--input-hours ({input_hours}) must be divisible by --n-subwindows ({n_subwindows})")
    if sw_features is None:
        sw_features = SW_FEATURES

    origin_index = decay.index
    parts = []

    def _sample(frame: pd.DataFrame) -> pd.DataFrame:
        pos = sw_minute.index.get_indexer(origin_index)
        valid = pos >= 0
        sub = frame.iloc[pos[valid]].copy()
        sub.index = origin_index[valid]
        return sub.reindex(origin_index)

    def _window_stats(hours: int, offset_hours: int = 0, label: str | None = None) -> pd.DataFrame:
        w = hours * 60
        min_periods = max(1, w // 4)
        source = sw_minute[sw_features]
        if offset_hours > 0:
            source = source.shift(offset_hours * 60)
        roll = source.rolling(window=w, min_periods=min_periods)
        roll_max = roll.max()
        roll_min = roll.min()
        suffix_label = label if label is not None else f"{hours}h"
        cols = [roll_max.add_suffix(f"_max_{suffix_label}"), roll_min.add_suffix(f"_min_{suffix_label}")]
        if window_stats == "richer":
            # Beyond the extremes, encode the window's central tendency and
            # spread directly -- min/max alone can't distinguish a channel
            # that sat high the whole window from one that spiked briefly
            # then fell, which matters for how "used up" the driving is.
            cols.append(roll.mean().add_suffix(f"_mean_{suffix_label}"))
            cols.append(roll.std().add_suffix(f"_std_{suffix_label}"))
        if include_relax_ratio:
            # (recent max - now) / (recent max - recent min): 0 = value is
            # still at its recent peak (driving ongoing), 1 = fully relaxed
            # back to the window's floor. Encodes "how depleted is the
            # driving relative to its recent peak" directly, instead of
            # making the model infer it from raw max/min levels -- motivated
            # by Stage 1 (momentum) being structurally blind to storm
            # recoveries (right before a recovery, recent momentum
            # necessarily still looks like decline) and Stage 2 needing an
            # explicit "this is over" signal solar wind can plausibly carry.
            # NOTE: "source" here is the same literal instantaneous minute
            # reading as "_now" below (just consumed as a ratio numerator
            # instead of standalone) -- so this is exactly as fragile to a
            # DSCOVR gap at the origin minute as "_now" is, and enabling it
            # drops the same rows to NaN. It is NOT a way to get the "relax"
            # signal without the "_now" NaN cost.
            span = roll_max - roll_min
            relax = ((roll_max - source) / span.where(span > 1e-9)).clip(lower=0.0, upper=1.0)
            cols.append(relax.add_suffix(f"_relax_{suffix_label}"))
        return _sample(pd.concat(cols, axis=1))

    def _shock_stats(hours: int) -> pd.DataFrame:
        # Biggest shock_hours-scale jump seen ANYWHERE in the trailing
        # `hours` window -- the 8h max/min/mean/std/relax stats above
        # characterize sustained driving level, but can dilute or miss an
        # abrupt trigger (a sudden Bz southward turn, a shock-front jump in
        # speed/density) that only lasts an hour or two. This is the
        # closest analog to the actual physical onset signature, motivated
        # by the "storm onset" showcase examples in plot_test_segments
        # needing solar wind to carry a signal Stage 1/momentum structurally
        # can't see yet.
        step = shock_hours * 60
        diff = sw_minute[sw_features].diff(step)
        w = hours * 60
        min_periods = max(1, w // 4)
        shock_max = diff.rolling(window=w, min_periods=min_periods).max()
        shock_min = diff.rolling(window=w, min_periods=min_periods).min()
        suffix = f"{hours}h"
        cols = [
            shock_max.add_suffix(f"_shock{shock_hours}h_max_{suffix}"),
            shock_min.add_suffix(f"_shock{shock_hours}h_min_{suffix}"),
        ]
        return _sample(pd.concat(cols, axis=1))

    if lag_features in ("both", "window"):
        if n_subwindows > 1:
            chunk = input_hours // n_subwindows
            for k in range(n_subwindows):
                parts.append(_window_stats(chunk, offset_hours=k * chunk, label=f"{chunk}h_chunk{k}"))
        else:
            parts.append(_window_stats(input_hours))
        if short_window_hours > 0:
            parts.append(_window_stats(short_window_hours))
        if include_now_feature:
            # Literal instantaneous reading (not an aggregate over a window)
            # -- the cleanest possible "what is driving doing RIGHT NOW"
            # signal, needed alongside the relax ratios above.
            parts.append(_sample(sw_minute[sw_features]).add_suffix("_now"))
        if shock_features:
            parts.append(_shock_stats(input_hours))

    if lag_features in ("both", "lags"):
        hourly = sw_minute[sw_features].resample("h").last()
        lag_frame = pd.concat(
            [hourly.shift(lag).add_suffix(f"_lag_{lag}h") for lag in range(input_hours)], axis=1,
        )
        parts.append(lag_frame.reindex(origin_index))

    if trend_hours > 0:
        hourly_now = sw_minute[sw_features].resample("h").last()
        trend = (hourly_now - hourly_now.shift(trend_hours)).add_suffix(f"_trend_{trend_hours}h")
        parts.append(trend.reindex(origin_index))

    sw_feats_hourly = pd.concat(parts, axis=1)

    decay_feats = pd.DataFrame(index=origin_index)
    decay_feats["decay_level"] = decay.reindex(origin_index).to_numpy()
    d = decay.reindex(origin_index)
    for h in decay_trend_hours:
        decay_feats[f"decay_change_{h}h"] = d.to_numpy() - d.shift(h).to_numpy()

    return sw_feats_hourly.join(decay_feats, how="inner")


def build_xy(features: pd.DataFrame, decay: pd.Series, out_hours: int):
    """Y(t, h) = decay(t+h) - decay(t) for h=1..out_hours (a whole window of
    horizons per origin, not one scalar). Rows with any NaN feature or any
    NaN target anywhere in the horizon window are dropped."""
    y = decay.to_numpy(dtype=np.float64)
    decay_times = decay.index
    pos_arr = decay_times.get_indexer(features.index)

    feat_arr = features.to_numpy(dtype=np.float64)
    feat_nan_mask = np.isnan(feat_arr).any(axis=1)

    n_decay = len(decay_times)
    X_rows, Y_rows, t_list, anchor_list = [], [], [], []
    for row_idx, i in enumerate(pos_arr):
        if i < 0 or i + out_hours >= n_decay or feat_nan_mask[row_idx]:
            continue
        anchor = y[i]
        future = y[i + 1:i + 1 + out_hours]
        if not np.isfinite(anchor) or not np.all(np.isfinite(future)):
            continue
        X_rows.append(feat_arr[row_idx])
        Y_rows.append(future - anchor)
        t_list.append(features.index[row_idx])
        anchor_list.append(anchor)

    if not X_rows:
        raise RuntimeError("No valid samples were built -- check data coverage/alignment.")
    return (
        np.stack(X_rows), np.stack(Y_rows),
        pd.DatetimeIndex(t_list), np.array(anchor_list, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Stage 2: shared horizon-conditioned LightGBM (from gbm_solar_wind_to_decay_v2.py
# -- horizon appended as a feature, one model shared across all horizons).
# ---------------------------------------------------------------------------

def _expand_horizons(X, Y, out_hours, target_mean, target_std, sample_weights=None):
    n = len(X)
    X_rep = np.repeat(X, out_hours, axis=0)
    horizon = np.tile(np.arange(1, out_hours + 1), n).astype(float)
    horizon_frac = horizon / float(out_hours)
    horizon_sq = horizon_frac ** 2
    X_long = np.column_stack([X_rep, horizon, horizon_frac, horizon_sq])
    Y_z = (Y - target_mean[None, :]) / target_std[None, :]
    y_long = Y_z.reshape(-1)
    w_long = np.repeat(sample_weights, out_hours) if sample_weights is not None else None
    return X_long, y_long, w_long


def _predict_shared(model, X, out_hours, target_mean, target_std):
    dummy = np.zeros((len(X), out_hours), dtype=float)
    X_long, _, _ = _expand_horizons(X, dummy, out_hours, target_mean, target_std)
    pred_z = model.predict(X_long, num_iteration=model.best_iteration).reshape(len(X), out_hours)
    return pred_z * target_std[None, :] + target_mean[None, :]


def train_shared_gbm(X_train, Y_train, X_val, Y_val, feature_names, out_hours,
                      params, num_boost_round, early_stopping_rounds, sample_weights):
    target_mean = np.nanmean(Y_train, axis=0)
    target_std = np.nanstd(Y_train, axis=0)
    target_std = np.where(target_std > 1e-8, target_std, 1.0)

    X_train_long, y_train_long, w_train_long = _expand_horizons(
        X_train, Y_train, out_hours, target_mean, target_std, sample_weights,
    )
    X_val_long, y_val_long, _ = _expand_horizons(X_val, Y_val, out_hours, target_mean, target_std)
    long_names = feature_names + ["forecast_horizon_h", "forecast_horizon_frac", "forecast_horizon_sq"]
    train_set = lgb.Dataset(X_train_long, label=y_train_long, weight=w_train_long, feature_name=long_names)
    val_set = lgb.Dataset(X_val_long, label=y_val_long, reference=train_set, feature_name=long_names)
    model = lgb.train(
        params, train_set, num_boost_round=num_boost_round,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(period=0)],
    )
    return model, target_mean, target_std, long_names


def fit_solar_wind_gate(pred_val: np.ndarray, residual_val: np.ndarray, max_gate: float) -> np.ndarray:
    """Fit a non-negative per-horizon scale factor from VALIDATION residuals:
    gate[h] = clip(dot(pred_val[:,h], residual_val[:,h]) / dot(pred_val[:,h], pred_val[:,h]), 0, max_gate).

    Replaces a single fixed scalar (the old --sw-weight) with one that can
    differ by horizon -- motivated by delta-correlation being roughly as
    good at 6-10h as at 1-4h while RMSE/R^2 still degrade there, which looks
    like a per-horizon AMPLITUDE/calibration problem (Stage 2's raw
    prediction is oversized or undersized at some horizons) rather than a
    directional one. This is the same fitting scheme gbm_solar_wind_to_decay_v2.py
    used -- and that file's docstring documents a real failure mode: if these
    gates are pinned at --max-gate across most horizons, that's a strong
    overfitting signal (the validation set wanted to amplify past the
    ceiling), not genuine skill. Watch the printed gate values for this.
    """
    out_hours = pred_val.shape[1]
    gates = np.zeros(out_hours, dtype=float)
    for h in range(out_hours):
        p = pred_val[:, h]
        y = residual_val[:, h]
        valid = np.isfinite(p) & np.isfinite(y)
        denom = float(np.dot(p[valid], p[valid])) if valid.any() else 0.0
        if denom <= 1e-12:
            gates[h] = 0.0
        else:
            gates[h] = np.clip(float(np.dot(p[valid], y[valid]) / denom), 0.0, max_gate)
    return gates


def fit_solar_wind_gate_asymmetric(
    pred_val: np.ndarray, residual_val: np.ndarray, max_gate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Like fit_solar_wind_gate(), but fits TWO scale factors per horizon --
    one applied when Stage 2's raw prediction is >= 0, another when it's < 0
    -- instead of a single symmetric scale.

    Motivated by a diagnosed directional bias: right before a storm recovery,
    Stage 1 (momentum) necessarily still extrapolates decline (that's what a
    trough looks like), and empirically Stage 2's RAW prediction already gets
    the sign right on these rows (positive, matching the true residual) but
    is drastically undersized relative to declines -- e.g. one measurement:
    raw prediction averaged +0.014 on rows where the true residual averaged
    +0.496, a ~35x undersizing, while the raw prediction on true-decline rows
    was already reasonably scaled. A single symmetric gate cannot fix both:
    amplifying enough to fix the increase-side blows up the decrease-side.
    Splitting by the raw prediction's own sign lets each direction get its
    own honest scale factor instead.

    Trade-off (measured): this substantially reduces the "predicts decrease
    when the actual outcome is an increase" rate (74-80% -> ~50%) at some
    cost to blanket RMSE at short horizons and a higher false-recovery rate.
    Deliberate: the whole point is to stop defaulting to "decrease" as the
    safe bet.
    """
    out_hours = pred_val.shape[1]
    gate_pos = np.zeros(out_hours, dtype=float)
    gate_neg = np.zeros(out_hours, dtype=float)
    for h in range(out_hours):
        p = pred_val[:, h]
        y = residual_val[:, h]
        valid = np.isfinite(p) & np.isfinite(y)
        for sel, gate_arr in ((valid & (p >= 0), gate_pos), (valid & (p < 0), gate_neg)):
            denom = float(np.dot(p[sel], p[sel])) if sel.any() else 0.0
            if denom <= 1e-12:
                gate_arr[h] = 0.0
            else:
                gate_arr[h] = np.clip(float(np.dot(p[sel], y[sel]) / denom), 0.0, max_gate)
    return gate_pos, gate_neg


def apply_asymmetric_gate(pred: np.ndarray, gate_pos: np.ndarray, gate_neg: np.ndarray) -> np.ndarray:
    return np.where(pred >= 0, pred * gate_pos[None, :], pred * gate_neg[None, :])


# ---------------------------------------------------------------------------
# Stormtime segments: whole-segment train/val/test bucket assignment
# ---------------------------------------------------------------------------

def load_stormtime_segments(path: str) -> pd.DataFrame:
    """Load find_stormtime_segments.py's CSV, sorted by start time."""
    df = pd.read_csv(path, parse_dates=["start", "end"]).sort_values("start").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No stormtime segments found in {path}")
    return df


OVERSIZED_SEGMENT_BUCKET = -2  # sentinel: split this segment's rows by date instead of keeping it whole


def assign_segment_buckets(
    segments: pd.DataFrame, val_date: pd.Timestamp, split_date: pd.Timestamp,
    max_segment_hours: float | None = None,
) -> np.ndarray:
    """0=train, 1=val, 2=test, decided by each segment's START time (whole
    segment, never split): keeps every row of a given storm in exactly one
    bucket instead of a plain per-row date cutoff slicing a storm in half.

    Segments longer than max_segment_hours get OVERSIZED_SEGMENT_BUCKET
    instead -- padding (e.g. +-24-36h) around frequent storms during an
    active period can chain many genuinely distinct events into one merged
    "segment" spanning months (observed: one 7009h/292-day segment, ~59% of
    all stormtime hours, versus a ~2000h/83-day segment as the next largest
    -- an order of magnitude gap). Such a segment isn't one event to protect
    from splitting; the caller should instead assign ITS rows individually by
    date, same as an ordinary chronological purge.
    """
    starts = segments["start"]
    bucket = np.where(starts < val_date, 0, np.where(starts < split_date, 1, 2))
    if max_segment_hours is not None:
        oversized = (segments["duration_hours"] > max_segment_hours).to_numpy()
        bucket = np.where(oversized, OVERSIZED_SEGMENT_BUCKET, bucket)
    return bucket


def segment_membership(times: pd.DatetimeIndex, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Index of the (sorted, non-overlapping) segment containing each timestamp, or -1."""
    t = np.asarray(times, dtype="datetime64[ns]")
    idx = np.searchsorted(starts, t, side="right") - 1
    idx = np.clip(idx, 0, len(starts) - 1)
    in_seg = (idx >= 0) & (t >= starts[idx]) & (t <= ends[idx])
    return np.where(in_seg, idx, -1)


# ---------------------------------------------------------------------------
# Predictive-power metrics, all computed PER HORIZON COLUMN
# ---------------------------------------------------------------------------

def r2_score(actual: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _by_horizon(fn, *arrays_2d) -> np.ndarray:
    """Apply a 2-arg metric fn column-by-column over (n, out_hours) arrays."""
    out_hours = arrays_2d[0].shape[1]
    return np.array([fn(*[a[:, h] for a in arrays_2d]) for h in range(out_hours)])


def rmse_1d(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Two-stage model: multi-output Ridge (momentum) + shared horizon-conditioned
# LightGBM (solar wind residual), trained ONLY on stormtime origins.
# ---------------------------------------------------------------------------

def train(args) -> None:
    val_date = pd.Timestamp(args.val_date)
    split_date = pd.Timestamp(args.split_date)
    if val_date >= split_date:
        raise ValueError("--val-date must precede --split-date")

    segments = load_stormtime_segments(args.segments_csv)
    if args.end_date:
        end_date = pd.Timestamp(args.end_date)
        before = len(segments)
        segments = segments[segments["start"] < end_date].reset_index(drop=True)
        print(f"--end-date={args.end_date}: dropped {before - len(segments)}/{before} segments starting "
              f"on/after it entirely (not reassigned to train -- excluded from every split). Use this to "
              f"cut off a stretch of degraded/incomplete DSCOVR data rather than letting --split-date's "
              f"unbounded 'everything after' land test there.")
    seg_bucket = assign_segment_buckets(segments, val_date, split_date, max_segment_hours=args.max_segment_hours)
    seg_starts = segments["start"].to_numpy()
    seg_ends = segments["end"].to_numpy()

    bucket_names = {0: "Train", 1: "Val", 2: "Test"}
    for b in (0, 1, 2):
        sel = segments[seg_bucket == b]
        span = f"{sel['start'].min()} to {sel['end'].max()}" if len(sel) else "n/a"
        print(f"{bucket_names[b]} segments: {len(sel)}/{len(segments)} "
              f"({sel['duration_hours'].sum():.0f}h total), {span}")
    oversized = segments[seg_bucket == OVERSIZED_SEGMENT_BUCKET]
    if len(oversized):
        print(f"Oversized segments (>{args.max_segment_hours:g}h -- likely many storms merged by padding "
              f"during an active period, not one event): {len(oversized)}/{len(segments)} "
              f"({oversized['duration_hours'].sum():.0f}h total). Their rows are assigned to "
              f"train/val/test individually by date below, NOT kept whole.")
        for _, row in oversized.iterrows():
            print(f"  segment {row['segment_id']}: {row['start']} to {row['end']} ({row['duration_hours']:.0f}h)")

    decay = build_decay_series(
        args.indir, args.satcat, args.object_type,
        args.weight_method, args.tau_minutes, args.min_gap_minutes,
        normalization_cutoff=val_date,
    )
    print(f"Decay series spans {decay.index.min()} to {decay.index.max()} ({len(decay)} hourly rows)")
    sw_minute = load_solar_wind_minute(args.dscovr_csv, max_fill_minutes=args.max_fill_minutes)
    print(f"Minute-resolution solar wind: {len(sw_minute)} rows, {sw_minute.index.min()} to {sw_minute.index.max()}")

    sw_features = SW_FEATURES if args.feature_set == "full" else TRIMMED_SW_FEATURES
    decay_trend_hours = decay_trend_hours_for(args.decay_history_hours)
    print(f"Engineering features from exactly the trailing {args.input_hours}h of solar wind "
          f"(--lag-features={args.lag_features}, --feature-set={args.feature_set} "
          f"[{len(sw_features)} channels], --window-stats={args.window_stats}) and "
          f"{args.decay_history_hours}h of decay history "
          f"(trend lags={decay_trend_hours}), target = a shared window of 1..{args.out_hours}h ahead...")
    features = build_features(
        sw_minute, decay, input_hours=args.input_hours,
        decay_history_hours=args.decay_history_hours, lag_features=args.lag_features,
        decay_trend_hours=decay_trend_hours,
        short_window_hours=args.short_window_hours, trend_hours=args.trend_hours,
        n_subwindows=args.n_subwindows, sw_features=sw_features,
        include_relax_ratio=args.relax_ratio_features,
        include_now_feature=args.now_feature,
        window_stats=args.window_stats,
        shock_features=args.shock_features, shock_hours=args.shock_hours,
    )
    feature_names = list(features.columns)
    decay_cols = [c for c in feature_names if c.startswith("decay_")]
    sw_cols = [c for c in feature_names if c not in decay_cols]
    print(f"Built {len(feature_names)} features ({len(sw_cols)} solar-wind, {len(decay_cols)} decay-momentum)")

    X, Y, sample_times, anchor = build_xy(features, decay, out_hours=args.out_hours)
    print(f"Built {len(X)} candidate hourly origins x {args.out_hours} horizons (before stormtime filtering)")

    seg_idx = segment_membership(sample_times, seg_starts, seg_ends)
    in_storm = seg_idx >= 0
    row_bucket = np.full(len(sample_times), -1, dtype=int)
    row_bucket[in_storm] = seg_bucket[seg_idx[in_storm]]

    target_end = sample_times + pd.to_timedelta(args.out_hours, unit="h")

    # Rows inside an OVERSIZED segment (see assign_segment_buckets) are
    # assigned individually by date instead of kept whole -- such a segment
    # is really a chain of many distinct storms merged by short quiet gaps,
    # not one event whose integrity needs protecting from a mid-split.
    oversized_mask = row_bucket == OVERSIZED_SEGMENT_BUCKET
    if oversized_mask.any():
        row_bucket[oversized_mask & (target_end < val_date)] = 0
        row_bucket[oversized_mask & (sample_times >= val_date) & (target_end < split_date)] = 1
        row_bucket[oversized_mask & (sample_times >= split_date)] = 2
        still_unassigned = row_bucket == OVERSIZED_SEGMENT_BUCKET
        row_bucket[still_unassigned] = -1  # shouldn't occur; safety net
        print(f"Reassigned {int(oversized_mask.sum())} hourly origins from oversized segment(s) "
              f"individually by date (train={int(np.sum(row_bucket[oversized_mask] == 0))}, "
              f"val={int(np.sum(row_bucket[oversized_mask] == 1))}, "
              f"test={int(np.sum(row_bucket[oversized_mask] == 2))}).")

    # Test ALWAYS stays restricted to the held-out storm segments -- that's
    # what we actually care about evaluating, regardless of --train-on-all-hours.
    test_mask = row_bucket == 2

    if args.train_on_all_hours:
        # Train/val use every chronological hour (not just the ~40% inside a
        # padded stormtime segment), purged by the same --val-date/--split-date
        # cutoffs the segment-level split already uses. Motivated by two
        # architecture changes (joint model, Stage 2 seeing momentum features)
        # both overfitting on the ~9-11k stormtime-only training rows -- more
        # turning-point examples, not more model flexibility, may be the real
        # lever. Test segments are chronologically after split_date by
        # construction, so this purge can't leak them into train/val.
        train_mask = target_end < val_date
        val_mask = (sample_times >= val_date) & (target_end < split_date)
        print(
            f"--train-on-all-hours: train/val use every hour (purged by --val-date/--split-date), "
            f"not just the {int(in_storm.sum())}/{len(in_storm)} ({100.0 * in_storm.mean():.1f}%) "
            f"hours inside a stormtime segment. train={int(train_mask.sum())}, val={int(val_mask.sum())}, "
            f"test={int(test_mask.sum())} (test unchanged: held-out storm segments only)."
        )
    else:
        train_mask = row_bucket == 0
        val_mask = row_bucket == 1
        print(
            f"Stormtime filter: {int(in_storm.sum())}/{len(in_storm)} hourly origins "
            f"({100.0 * in_storm.mean():.1f}%) fall inside a stormtime segment. "
            f"Segment-level split: train={int(train_mask.sum())}, val={int(val_mask.sum())}, "
            f"test={int(test_mask.sum())} (whole segments, never split mid-segment; "
            f"non-stormtime hours excluded from all three)."
        )
    if min(train_mask.sum(), val_mask.sum(), test_mask.sum()) < 20:
        raise ValueError(
            f"Too few samples after the split: train={train_mask.sum()}, "
            f"val={val_mask.sum()}, test={test_mask.sum()}. Adjust --val-date/--split-date or widen "
            f"--pad-hours in find_stormtime_segments.py."
        )

    X_train, Y_train = X[train_mask], Y[train_mask]
    X_val, Y_val = X[val_mask], Y[val_mask]
    X_test, Y_test = X[test_mask], Y[test_mask]
    anchor_test = anchor[test_mask]
    test_times = sample_times[test_mask]
    test_seg_idx = seg_idx[test_mask]

    decay_idx = [feature_names.index(c) for c in decay_cols]
    sw_idx = [feature_names.index(c) for c in sw_cols]
    X_decay_train, X_decay_val, X_decay_test = X_train[:, decay_idx], X_val[:, decay_idx], X_test[:, decay_idx]

    # Stage 2's OWN feature set: "all" (default) gives it decay-momentum
    # features too, not just solar wind, so it can learn interactions like
    # "momentum says decline but driving has relaxed -> predict a positive
    # residual" as a tree split -- while Stage 1's Ridge prediction still
    # provides the additive floor (unlike --architecture joint, which drops
    # that floor entirely and was found to overfit badly: best iteration
    # 1303, RMSE 34-91% WORSE than persistence). "solar-wind" reproduces the
    # original solar-wind-only Stage 2.
    if args.stage2_features == "all":
        stage2_cols, X_stage2_train, X_stage2_val, X_stage2_test = feature_names, X_train, X_val, X_test
    else:
        stage2_cols, X_stage2_train, X_stage2_val, X_stage2_test = sw_cols, X_train[:, sw_idx], X_val[:, sw_idx], X_test[:, sw_idx]

    print(f"\nStage 1 (baseline, always fit for comparison): ridge on {len(decay_cols)} decay-momentum "
          f"feature(s), {args.out_hours} outputs at once (alpha={args.stage1_ridge_alpha})")
    stage1 = Ridge(alpha=args.stage1_ridge_alpha).fit(X_decay_train, Y_train)
    stage1_pred_test = stage1.predict(X_decay_test)

    # Event weights from the worst horizon per origin (matches
    # gbm_solar_wind_to_decay_v2.py's multi-horizon event-weighting).
    train_severity = np.max(np.abs(Y_train), axis=1)
    event_threshold = float(np.quantile(train_severity, args.event_quantile))
    train_weight = np.ones(len(Y_train), dtype=np.float64)
    train_weight[train_severity >= event_threshold] = args.event_weight
    print(f"Event weighting: threshold={event_threshold:.3f} (train {args.event_quantile:.0%} quantile "
          f"of worst-horizon |change|), {np.mean(train_severity >= event_threshold) * 100:.1f}% of "
          f"training origins weighted x{args.event_weight:g}")

    # Onset weighting: rows where Stage 1's OWN prediction is ~flat (momentum
    # hasn't picked up on anything) get extra weight (stacks multiplicatively
    # with event weighting above), so the GBM is explicitly pushed to get
    # good at the rare "solar wind must carry the whole signal" case instead
    # of being dominated by the much more common mid-storm refinement rows.
    stage1_pred_train = stage1.predict(X_decay_train)
    train_onset_mask = np.abs(stage1_pred_train[:, -1]) < args.onset_momentum_threshold
    if args.onset_weight != 1.0:
        train_weight[train_onset_mask] *= args.onset_weight
        print(f"Onset weighting: {np.mean(train_onset_mask) * 100:.1f}% of training origins have Stage 1's "
              f"own final-horizon |predicted change| < {args.onset_momentum_threshold:g} "
              f"(momentum ~flat), weighted x{args.onset_weight:g}")

    params = {
        "objective": args.objective, "metric": "rmse", "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves, "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
        "lambda_l1": 0.1, "lambda_l2": 0.1, "verbosity": -1, "seed": args.seed,
    }
    if args.objective == "huber":
        params["alpha"] = args.huber_alpha

    horizons = np.arange(1, args.out_hours + 1)
    lo_q, hi_q = args.band_quantiles
    quantile_params = dict(params)
    quantile_params["objective"] = "quantile"
    quantile_params.pop("alpha", None)

    if args.architecture == "joint":
        # ONE shared horizon-conditioned LightGBM sees decay-momentum AND
        # solar-wind features TOGETHER and predicts Y directly (not a residual
        # on top of Stage 1) -- so it can learn INTERACTIONS an additive
        # two-stage split cannot represent, e.g. "momentum says decline but
        # the relaxation-ratio features say driving has stopped -> predict
        # recovery instead." Stage 1 above is still fit purely as a reported
        # baseline, not part of this prediction.
        print(f"Architecture: JOINT -- one shared horizon-conditioned LightGBM on all "
              f"{len(feature_names)} features (decay-momentum + solar-wind) together, "
              f"objective={args.objective}, trained ONLY on {len(X_train)} stormtime training origins...")
        model, target_mean, target_std, long_feature_names = train_shared_gbm(
            X_train, Y_train, X_val, Y_val, feature_names, args.out_hours,
            params, args.num_boost_round, args.early_stopping_rounds, train_weight,
        )
        print(f"Best boosting round: {model.best_iteration}")
        pred_delta = _predict_shared(model, X_test, args.out_hours, target_mean, target_std)
        gate_pos = gate_neg = np.ones(args.out_hours, dtype=float)  # not used; kept for table/JSON shape

        print(f"Training uncertainty band: LightGBM quantile models at alpha={lo_q:g} and {hi_q:g} "
              f"(same joint feature set, target Y directly)...")
        model_lo, mean_lo, std_lo, _ = train_shared_gbm(
            X_train, Y_train, X_val, Y_val, feature_names, args.out_hours,
            dict(quantile_params, alpha=lo_q), args.num_boost_round, args.early_stopping_rounds, train_weight,
        )
        model_hi, mean_hi, std_hi, _ = train_shared_gbm(
            X_train, Y_train, X_val, Y_val, feature_names, args.out_hours,
            dict(quantile_params, alpha=hi_q), args.num_boost_round, args.early_stopping_rounds, train_weight,
        )
        pred_lo = _predict_shared(model_lo, X_test, args.out_hours, mean_lo, std_lo)
        pred_hi = _predict_shared(model_hi, X_test, args.out_hours, mean_hi, std_hi)

        # Not meaningful under the joint architecture -- there's no separate
        # "solar wind alone" residual model to isolate; leave NaN rather than
        # report a number that doesn't mean what it used to.
        delta_corr_sw_isolated_h = np.full(args.out_hours, np.nan)
        r2_sw_isolated_h = np.full(args.out_hours, np.nan)
        importance_index_filter = set(feature_names)  # rank decay + solar-wind features together
    else:
        stage1_pred_train = stage1.predict(X_decay_train)
        stage1_pred_val = stage1.predict(X_decay_val)
        gbm_train_target = Y_train - stage1_pred_train
        gbm_val_target = Y_val - stage1_pred_val

        print(f"Architecture: TWO-STAGE -- shared horizon-conditioned LightGBM on {len(stage2_cols)} "
              f"feature(s) ({args.stage2_features}) x {args.out_hours} horizons, objective={args.objective}, "
              f"trained ONLY on {len(X_stage2_train)} stormtime training origins...")
        model, target_mean, target_std, long_feature_names = train_shared_gbm(
            X_stage2_train, gbm_train_target, X_stage2_val, gbm_val_target, stage2_cols, args.out_hours,
            params, args.num_boost_round, args.early_stopping_rounds, train_weight,
        )
        print(f"Best boosting round: {model.best_iteration}")

        gbm_pred_val = _predict_shared(model, X_stage2_val, args.out_hours, target_mean, target_std)
        gbm_pred_test = _predict_shared(model, X_stage2_test, args.out_hours, target_mean, target_std)

        if args.asymmetric_gate:
            gate_pos, gate_neg = fit_solar_wind_gate_asymmetric(gbm_pred_val, gbm_val_target, max_gate=args.max_gate)
            print("Solar-wind validation gates by horizon (ASYMMETRIC: separate scale for raw prediction "
                  ">=0 vs <0 -- see fit_solar_wind_gate_asymmetric()'s docstring):")
            print("  when raw pred >= 0 (predicting recovery): "
                  + "  ".join(f"h{h + 1}={g:.2f}" for h, g in enumerate(gate_pos)))
            print("  when raw pred <  0 (predicting decline):  "
                  + "  ".join(f"h{h + 1}={g:.2f}" for h, g in enumerate(gate_neg)))
            for label, arr in (("positive", gate_pos), ("negative", gate_neg)):
                if np.any(arr >= args.max_gate - 1e-9):
                    pinned = [h + 1 for h, g in enumerate(arr) if g >= args.max_gate - 1e-9]
                    print(f"  WARNING: {label}-side gate(s) pinned at --max-gate={args.max_gate:g} for "
                          f"horizon(s) {pinned} -- likely overfitting signal, not confirmed genuine skill.")
            pred_delta = stage1_pred_test + apply_asymmetric_gate(gbm_pred_test, gate_pos, gate_neg)
        else:
            gates = fit_solar_wind_gate(gbm_pred_val, gbm_val_target, max_gate=args.max_gate)
            print("Solar-wind validation gates by horizon (per-horizon calibration, replaces a single "
                  "fixed --sw-weight):")
            print("  " + "  ".join(f"h{h + 1}={g:.2f}" for h, g in enumerate(gates)))
            if np.any(gates >= args.max_gate - 1e-9):
                pinned = [h + 1 for h, g in enumerate(gates) if g >= args.max_gate - 1e-9]
                print(f"  WARNING: gate(s) pinned at --max-gate={args.max_gate:g} for horizon(s) {pinned} "
                      f"-- per fit_solar_wind_gate()'s docstring, this is a likely overfitting signal on "
                      f"the small validation set, not confirmed genuine skill. Consider lowering --max-gate.")
            gate_pos = gate_neg = gates
            pred_delta = stage1_pred_test + gates[None, :] * gbm_pred_test

        print(f"Training uncertainty band: LightGBM quantile models at alpha={lo_q:g} and {hi_q:g} "
              f"(same Stage-1 residual target, same feature set)...")
        model_lo, mean_lo, std_lo, _ = train_shared_gbm(
            X_stage2_train, gbm_train_target, X_stage2_val, gbm_val_target, stage2_cols, args.out_hours,
            dict(quantile_params, alpha=lo_q), args.num_boost_round, args.early_stopping_rounds, train_weight,
        )
        model_hi, mean_hi, std_hi, _ = train_shared_gbm(
            X_stage2_train, gbm_train_target, X_stage2_val, gbm_val_target, stage2_cols, args.out_hours,
            dict(quantile_params, alpha=hi_q), args.num_boost_round, args.early_stopping_rounds, train_weight,
        )
        # NOT scaled by the point model's per-horizon gate: the gate answers "how much
        # should the MEAN correction be trusted" and correctly shrinks toward 0 at long
        # horizons to avoid overfitting -- but that says nothing about how UNCERTAIN the
        # outcome is. Scaling the band by it too collapsed band width to ~0 exactly at
        # the horizons where a band is most needed (empirically: coverage fell from a
        # sane ~70-80% at h1-4 to 2% at h9 and 0% at h10 when this was tried). Pinball
        # loss already calibrates each quantile model on its own; use it directly.
        pred_lo = stage1_pred_test + _predict_shared(model_lo, X_stage2_test, args.out_hours, mean_lo, std_lo)
        pred_hi = stage1_pred_test + _predict_shared(model_hi, X_stage2_test, args.out_hours, mean_hi, std_hi)

        residual_actual = Y_test - stage1_pred_test  # what Stage 1/momentum couldn't explain
        residual_pred = gbm_pred_test                # Stage 2's (unweighted) guess at that residual
        delta_corr_sw_isolated_h = _by_horizon(corr_1d, residual_pred, residual_actual)
        r2_sw_isolated_h = _by_horizon(r2_score, residual_actual, residual_pred)
        importance_index_filter = set(stage2_cols)

    actual_abs = anchor_test[:, None] + Y_test
    combined_abs = anchor_test[:, None] + pred_delta
    stage1_abs = anchor_test[:, None] + stage1_pred_test
    persistence_abs = np.repeat(anchor_test[:, None], args.out_hours, axis=1)

    combined_lo_abs = anchor_test[:, None] + pred_lo
    combined_hi_abs = anchor_test[:, None] + pred_hi
    # Quantile crossing (lo > hi) is possible when two independently-trained
    # quantile models disagree; enforce the band is a real interval.
    combined_lo_abs, combined_hi_abs = (
        np.minimum(combined_lo_abs, combined_hi_abs), np.maximum(combined_lo_abs, combined_hi_abs),
    )

    band_coverage_h = np.mean((actual_abs >= combined_lo_abs) & (actual_abs <= combined_hi_abs), axis=0)
    nominal_coverage = hi_q - lo_q
    print(f"Band coverage by horizon (nominal target {nominal_coverage:.0%} for "
          f"[{lo_q:g}, {hi_q:g}] quantiles): "
          + ", ".join(f"h{h}={c:.0%}" for h, c in zip(horizons, band_coverage_h)))

    rmse_combined_h = _by_horizon(rmse_1d, actual_abs, combined_abs)
    rmse_stage1_h = _by_horizon(rmse_1d, actual_abs, stage1_abs)
    rmse_persist_h = _by_horizon(rmse_1d, actual_abs, persistence_abs)

    r2_combined_h = _by_horizon(r2_score, actual_abs, combined_abs)
    r2_stage1_h = _by_horizon(r2_score, actual_abs, stage1_abs)
    r2_persist_h = _by_horizon(r2_score, actual_abs, persistence_abs)

    delta_corr_combined_h = _by_horizon(corr_1d, pred_delta, Y_test)
    delta_corr_stage1_h = _by_horizon(corr_1d, stage1_pred_test, Y_test)

    print(f"\nStormtime test metrics by horizon (n={len(actual_abs)} test origins):")
    table = pd.DataFrame({
        "horizon_h": horizons,
        "gate_pos": gate_pos,
        "gate_neg": gate_neg,
        "rmse_persistence": rmse_persist_h, "rmse_stage1": rmse_stage1_h, "rmse_combined": rmse_combined_h,
        "skill_vs_persist_%": (1 - rmse_combined_h / rmse_persist_h) * 100,
        "skill_vs_stage1_%": (1 - rmse_combined_h / rmse_stage1_h) * 100,
        "r2_persistence": r2_persist_h, "r2_stage1": r2_stage1_h, "r2_combined": r2_combined_h,
        "delta_corr_stage1": delta_corr_stage1_h, "delta_corr_sw_isolated": delta_corr_sw_isolated_h,
        "delta_corr_combined": delta_corr_combined_h,
    })
    print(table.to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    mask = actual_abs < args.storm_threshold
    if mask.any():
        ev_combined = rmse_1d(actual_abs[mask], combined_abs[mask])
        ev_stage1 = rmse_1d(actual_abs[mask], stage1_abs[mask])
        ev_persist = rmse_1d(actual_abs[mask], persistence_abs[mask])
        print(f"\nCore-storm slice (dsn < {args.storm_threshold:g}, n={int(mask.sum())}/{mask.size} "
              f"of the already-stormtime-only test set, pooled across all horizons): "
              f"combined={ev_combined:.4f}  Stage1={ev_stage1:.4f}  persistence={ev_persist:.4f}  "
              f"(combined vs. persistence {(1 - ev_combined / ev_persist) * 100:+.1f}%, "
              f"vs. Stage1 {(1 - ev_combined / ev_stage1) * 100:+.1f}%)")
    else:
        print(f"\nCore-storm slice (dsn < {args.storm_threshold:g}): no test samples met this threshold")

    # Onset-only slice: origins where Stage 1's own final-horizon predicted
    # change was ~flat (momentum hadn't picked up on anything yet). This is
    # an honest, separate look at whether solar wind is adding ANTICIPATORY
    # skill, not just refining a trend momentum already caught -- the
    # blanket metrics above are dominated by mid-storm rows (segments are
    # built by padding around an ALREADY-detected decay threshold crossing,
    # see find_stormtime_segments.py), so they can look fine even if the
    # model never adds value at the moment it would matter most.
    onset_row_mask = np.abs(stage1_pred_test[:, -1]) < args.onset_momentum_threshold
    onset_skill_vs_stage1_h = None
    if onset_row_mask.any():
        rmse_combined_onset_h = _by_horizon(rmse_1d, actual_abs[onset_row_mask], combined_abs[onset_row_mask])
        rmse_stage1_onset_h = _by_horizon(rmse_1d, actual_abs[onset_row_mask], stage1_abs[onset_row_mask])
        rmse_persist_onset_h = _by_horizon(rmse_1d, actual_abs[onset_row_mask], persistence_abs[onset_row_mask])
        onset_skill_vs_stage1_h = (1 - rmse_combined_onset_h / rmse_stage1_onset_h) * 100
        onset_skill_vs_persist_h = (1 - rmse_combined_onset_h / rmse_persist_onset_h) * 100
        print(f"\nOnset-only slice (Stage 1's own final-horizon |predicted change| < "
              f"{args.onset_momentum_threshold:g}, n={int(onset_row_mask.sum())}/{len(onset_row_mask)} "
              f"test origins where momentum was ~flat):")
        print("  skill_vs_stage1_%: " + "  ".join(f"h{h}={s:+.1f}" for h, s in zip(horizons, onset_skill_vs_stage1_h)))
        print("  skill_vs_persist_%: " + "  ".join(f"h{h}={s:+.1f}" for h, s in zip(horizons, onset_skill_vs_persist_h)))
        print(f"  mean skill_vs_stage1={np.mean(onset_skill_vs_stage1_h):+.1f}%, "
              f"mean skill_vs_persist={np.mean(onset_skill_vs_persist_h):+.1f}%")
    else:
        print(f"\nOnset-only slice: no test origins had Stage 1 |predicted change| < "
              f"{args.onset_momentum_threshold:g}")

    print(f"\nPredictive power of solar wind (averaged across all {args.out_hours} horizons):")
    print(f"  R^2 gain from adding solar wind (combined - Stage1): {np.mean(r2_combined_h - r2_stage1_h):+.4f}")
    print(f"  Solar wind isolated R^2 on the Stage-1 residual:     {np.mean(r2_sw_isolated_h):.4f}")
    print(f"  Delta-corr, combined:       {np.mean(delta_corr_combined_h):.3f} "
          f"(r^2={np.mean(delta_corr_combined_h ** 2):.4f})")
    print(f"  Delta-corr, solar wind only: {np.mean(delta_corr_sw_isolated_h):.3f} "
          f"(r^2={np.mean(delta_corr_sw_isolated_h ** 2):.4f})")

    imp = pd.Series(model.feature_importance(importance_type="gain"), index=long_feature_names)
    imp = imp[imp.index.isin(importance_index_filter)]  # drop the 3 horizon meta-features from the ranking
    imp = (imp / imp.sum()).sort_values(ascending=False)
    what = "solar-wind and decay-momentum" if args.architecture == "joint" else "solar-wind"
    print(f"\nTop {args.top_features} {what} features by gain importance:")
    for name, val in imp.head(args.top_features).items():
        print(f"  {prettify_engineered_feature_name(name):55s} {val:.4f}")

    metrics_by_horizon = table
    if args.save_model:
        out_dir = Path(args.save_model)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(str(out_dir / "stage2_gbm.txt"))
        metrics_by_horizon.to_csv(out_dir / "metrics_by_horizon.csv", index=False)
        summary = {
            "out_hours": args.out_hours,
            "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
            "max_gate": args.max_gate,
            "architecture": args.architecture,
            "asymmetric_gate": args.asymmetric_gate,
            "gate_pos_by_horizon": [float(g) for g in gate_pos],
            "gate_neg_by_horizon": [float(g) for g in gate_neg],
            "mean_r2_gain_from_solar_wind": float(np.mean(r2_combined_h - r2_stage1_h)),
            "mean_r2_solar_wind_isolated": float(np.mean(r2_sw_isolated_h)),
            "mean_delta_corr_combined": float(np.mean(delta_corr_combined_h)),
            "mean_delta_corr_sw_isolated": float(np.mean(delta_corr_sw_isolated_h)),
            "band_quantiles": list(args.band_quantiles),
            "band_coverage_by_horizon": [float(c) for c in band_coverage_h],
            "band_coverage_nominal": float(nominal_coverage),
            "onset_momentum_threshold": args.onset_momentum_threshold,
            "onset_n_test_origins": int(onset_row_mask.sum()),
            "onset_skill_vs_stage1_by_horizon": (
                [float(s) for s in onset_skill_vs_stage1_h] if onset_skill_vs_stage1_h is not None else None
            ),
        }
        model_lo.save_model(str(out_dir / "stage2_gbm_lo.txt"))
        model_hi.save_model(str(out_dir / "stage2_gbm_hi.txt"))
        with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved model and metrics to {out_dir}")

    if args.plot:
        plot_horizon_diagnostics(
            horizons, rmse_combined_h, rmse_stage1_h, rmse_persist_h,
            r2_combined_h, r2_stage1_h, r2_persist_h, r2_sw_isolated_h,
            delta_corr_combined_h, delta_corr_stage1_h, delta_corr_sw_isolated_h,
            imp.head(args.top_features), args.plot,
        )

    if args.zoom_plot:
        plot_test_segments(decay, test_times, test_seg_idx, segments,
                            anchor_test, combined_abs, combined_lo_abs, combined_hi_abs, stage1_abs,
                            args.out_hours, args.zoom_plot,
                            margin_hours=args.zoom_margin_hours,
                            fan_stride_hours=args.zoom_fan_stride_hours,
                            max_segments=args.zoom_max_segments,
                            max_fans_per_segment=args.zoom_max_fans,
                            min_change=args.zoom_min_change,
                            min_magnitude_ratio=args.zoom_min_magnitude_ratio,
                            onset_momentum_threshold=args.zoom_onset_momentum_threshold,
                            min_early_divergence_ratio=args.zoom_min_early_divergence_ratio,
                            quiet_lead_hours=args.zoom_quiet_lead_hours,
                            max_quiet_actual_ratio=args.zoom_max_quiet_actual_ratio,
                            max_quiet_pred_err=args.zoom_max_quiet_pred_err)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_WINDOW_SUFFIX_RE = re.compile(r",?\s*\d+h window")


def _strip_window_suffix(label: str) -> str:
    """Drop the "8h window" clause from a polished feature label (e.g.
    "IMF By (GSM) -- min, 8h window (part 1)" -> "IMF By (GSM) -- min (part
    1)") for the feature-importance panel: with --n-subwindows fixed, every
    label repeats the same window size, so it's pure clutter there."""
    return _WINDOW_SUFFIX_RE.sub("", label)


def _style_axis(ax) -> None:
    """Shared look: full black box border on all four sides (matching the
    L1 solar-wind plot), light horizontal grid only."""
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)


def plot_horizon_diagnostics(
    horizons: np.ndarray,
    rmse_combined_h, rmse_stage1_h, rmse_persist_h,
    r2_combined_h, r2_stage1_h, r2_persist_h, r2_sw_isolated_h,
    delta_corr_combined_h, delta_corr_stage1_h, delta_corr_sw_isolated_h,
    importances: pd.Series | None,
    output_path: str,
) -> None:
    """RMSE / R^2 / delta-correlation, each vs. horizon, each comparing
    Combined/Stage 1/Persistence, plus feature importances (skipped if
    `importances` is None -- not every backend has a gain-importance
    equivalent, e.g. an LSTM)."""
    n_panels = 4 if importances is not None else 3
    plt.rcParams["font.size"] = 13
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 4.6 * n_panels))
    fig.patch.set_facecolor("white")

    # --- Panel 1: RMSE by horizon, explicitly compared to Persistence ------
    ax = axes[0]
    ax.plot(horizons, rmse_persist_h, color="0.55", linestyle="--", marker="o", markersize=5,
            linewidth=1.5, label="Persistence")
    ax.plot(horizons, rmse_stage1_h, color="#6AA56D", marker="o", markersize=5,
            linewidth=1.8, label="Stage 1 (Momentum)")
    ax.plot(horizons, rmse_combined_h, color="tab:blue", marker="o", markersize=6,
            linewidth=2.4, label="Combined Forecast")
    ax.set_xlabel("Forecast Horizon (Hours Ahead)")
    ax.set_ylabel("RMSE")
    ax.set_xticks(horizons)
    ax.set_title("RMSE by Forecast Horizon", fontsize=13, fontweight="semibold")
    ax.legend(fontsize=11, loc="best", frameon=False)
    _style_axis(ax)

    # --- Panel 2: R^2 by horizon --------------------------------------------
    ax = axes[1]
    ax.plot(horizons, r2_persist_h, color="0.55", linestyle="--", marker="o", markersize=5,
            linewidth=1.5, label="Persistence")
    ax.plot(horizons, r2_stage1_h, color="#6AA56D", marker="o", markersize=5,
            linewidth=1.8, label="Stage 1 (Momentum)")
    ax.plot(horizons, r2_sw_isolated_h, color="0.25", marker="s", markersize=5,
            linewidth=2.0, label="Stage 2 (Solar Wind Alone)")
    ax.plot(horizons, r2_combined_h, color="tab:blue", marker="o", markersize=6,
            linewidth=2.4, label="Combined")
    ax.axhline(0, color="0.3", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Forecast Horizon (Hours Ahead)")
    ax.set_ylabel(r"$R^2$ vs. Actual")
    ax.set_xticks(horizons)
    ax.set_title(r"$R^2$ by Forecast Horizon", fontsize=13, fontweight="semibold")
    ax.legend(fontsize=11, loc="best", frameon=False)
    _style_axis(ax)

    # --- Panel 3: delta correlation by horizon, model-level only -----------
    ax = axes[2]
    ax.plot(horizons, delta_corr_stage1_h, color="#6AA56D", linewidth=2.0, marker="o", markersize=5,
            label="Stage 1 (Momentum)")
    ax.plot(horizons, delta_corr_sw_isolated_h, color="0.25", linewidth=2.0, marker="s", markersize=5,
            label="Stage 2 (Solar Wind Alone)")
    ax.plot(horizons, delta_corr_combined_h, color="tab:blue", linewidth=2.4, marker="^", markersize=6,
            label="Combined Forecast")
    ax.axhline(0, color="0.3", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Forecast Horizon (Hours Ahead)")
    ax.set_ylabel("Delta Correlation")
    ax.set_xticks(horizons)
    ax.set_title("Delta Correlation by Horizon", fontsize=13, fontweight="semibold")
    ax.legend(fontsize=11, loc="best", frameon=False)
    _style_axis(ax)

    # --- Panel 4: feature importance (only if provided) --------------------
    if importances is not None:
        ax = axes[3]
        imp_sorted = importances.sort_values(ascending=True)
        polished_labels = [_strip_window_suffix(prettify_engineered_feature_name(name))
                            for name in imp_sorted.index]
        bar_colors = plt.get_cmap("Blues")(np.linspace(0.45, 0.95, len(imp_sorted)))
        ax.barh(polished_labels, imp_sorted.to_numpy(), color=bar_colors)
        ax.set_xlabel("Normalized Gain Importance")
        ax.set_title(f"Top {len(importances)} Solar-Wind Features (Stage 2 Gain)",
                     fontsize=13, fontweight="semibold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)

    fig.tight_layout(h_pad=2.5)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved validation plot to {output_path}")


def plot_test_segments(decay, test_times, test_seg_idx, segments, anchor_test, combined_abs,
                        combined_lo_abs, combined_hi_abs, stage1_abs,
                        out_hours, output_path, margin_hours: float = 10.0,
                        fan_stride_hours: float = 4.0,
                        max_segments: int = 2, max_fans_per_segment: int = 1,
                        min_change: float = 0.3, min_magnitude_ratio: float = 0.5,
                        onset_momentum_threshold: float = 0.15,
                        min_early_divergence_ratio: float = 0.3,
                        quiet_lead_hours: int = 2, max_quiet_actual_ratio: float = 0.3,
                        max_quiet_pred_err: float = 0.15) -> None:
    """A curated showcase of examples where SOLAR WIND DEMONSTRABLY HELPED,
    not just a good-looking forecast: each panel is built around one
    forecast origin where (1) a real decay/increase happened (actual net
    change over 1..out_hours, |.| >= min_change), (2) the combined forecast
    called it correctly in both sign and magnitude (>= min_magnitude_ratio
    of the real move), (3) the combined forecast's final-horizon error is
    strictly lower than Stage 1 (momentum-only)'s -- i.e. adding solar wind
    actually beat pure persistence for this specific event, not just matched
    it, (4) combined had already committed to the call (right sign, >=
    min_early_divergence_ratio of the eventual move) by the HALFWAY horizon,
    not just at the very end -- otherwise a "hit" could just be persistence
    with a late correction tacked on near the final horizon, which isn't
    really an onset call -- AND (5) the actual value was still quiet (near
    its anchor) at quiet_lead_hours in, with combined ALSO tracking it
    closely there, so the panel visibly shows "matches reality while calm,
    then correctly breaks for the event" rather than diverging from the
    very first predicted hour regardless of whether the storm has started
    yet. Stage 1's own trajectory is drawn alongside (dashed green)
    so the reader can see exactly what momentum alone would have predicted
    vs. what combined solar-wind-informed forecast + actual did.

    Selection prioritizes STORM-ONSET examples first -- origins where
    Stage 1's own predicted change was small (|.| < onset_momentum_threshold,
    i.e. momentum was still ~flat, structurally blind to what's coming)
    while solar wind correctly called a real move anyway -- the clearest
    demonstration of solar wind adding information beyond persistence,
    rather than refining a trend momentum had already picked up. Within
    each tier (onset vs. not), ranks by biggest improvement over Stage 1,
    then by biggest actual move. Tries to include at least one decay and
    one increase example, and skips any candidate whose zoom window
    overlaps one already chosen, so shown examples are non-overlapping
    distinct events. If nothing qualifies, prints how many candidates
    passed each successive filter so you can see which condition is binding.

    NOTE: each row here is a DIFFERENT event on a DIFFERENT date range, so
    fig.autofmt_xdate() must NOT be used -- it assumes a shared x-axis
    across stacked subplots and hides tick labels on every row but the
    bottom one. Each axis gets its own rotated date ticks instead.
    """
    n = len(test_times)
    out_td = pd.Timedelta(hours=out_hours)

    # ---- Vectorized per-origin quality metrics across the WHOLE test set:
    # full-trajectory RMSE/band-coverage for combined AND Stage 1, and the
    # actual vs. predicted NET change over the whole horizon (the "did it
    # call a decay/increase" signal), all in out_hours passes instead of a
    # python loop per row.
    sq_err_sum = np.zeros(n)
    stage1_sq_err_sum = np.zeros(n)
    count = np.zeros(n)
    in_band_count = np.zeros(n)
    in_band_total = np.zeros(n)
    final_actual = np.full(n, np.nan)
    actual_matrix = np.full((n, out_hours), np.nan)  # actual value at each horizon, for the quiet-lead-in check
    for h in range(1, out_hours + 1):
        target_t = test_times + pd.Timedelta(hours=h)
        actual_h = decay.reindex(target_t).to_numpy()
        valid_h = np.isfinite(actual_h)
        actual_matrix[valid_h, h - 1] = actual_h[valid_h]
        pred_h = combined_abs[:, h - 1]
        sq_err_sum[valid_h] += (actual_h[valid_h] - pred_h[valid_h]) ** 2
        stage1_sq_err_sum[valid_h] += (actual_h[valid_h] - stage1_abs[valid_h, h - 1]) ** 2
        count[valid_h] += 1
        lo_h, hi_h = combined_lo_abs[:, h - 1], combined_hi_abs[:, h - 1]
        in_band = valid_h & (lo_h <= actual_h) & (actual_h <= hi_h)
        in_band_count += in_band.astype(int)
        in_band_total += valid_h.astype(int)
        if h == out_hours:
            final_actual = np.where(valid_h, actual_h, final_actual)

    with np.errstate(invalid="ignore", divide="ignore"):
        fan_rmse_all = np.sqrt(np.where(count > 0, sq_err_sum / np.maximum(count, 1), np.nan))
        stage1_rmse_all = np.sqrt(np.where(count > 0, stage1_sq_err_sum / np.maximum(count, 1), np.nan))
    fan_rmse_all[count == 0] = np.nan
    stage1_rmse_all[count == 0] = np.nan
    actual_change = final_actual - anchor_test
    pred_change = combined_abs[:, out_hours - 1] - anchor_test
    stage1_change = stage1_abs[:, out_hours - 1] - anchor_test
    # Full-trajectory RMSE (not just the final-horizon point) so the
    # "combined beat Stage 1" selection criterion matches exactly what the
    # title reports -- using final-horizon error alone here previously let a
    # candidate through that won at h10 but lost on average across h1-h10,
    # so its displayed RMSE comparison could read as a contradiction.
    improvement = stage1_rmse_all - fan_rmse_all  # > 0: solar wind beat Stage 1 on this event
    is_onset = np.abs(stage1_change) < onset_momentum_threshold

    # ---- Subsample per segment (>= fan_stride_hours apart) to avoid near-
    # duplicate adjacent origins dominating candidate selection.
    candidate_rows = []
    for seg_id in sorted(set(int(s) for s in test_seg_idx)):
        row_idx = np.flatnonzero(test_seg_idx == seg_id)
        row_idx = row_idx[np.argsort(test_times[row_idx].to_numpy())]
        kept = [0] if len(row_idx) else []
        for i in range(1, len(row_idx)):
            if (test_times[row_idx[i]] - test_times[row_idx[kept[-1]]]) >= pd.Timedelta(hours=fan_stride_hours):
                kept.append(i)
        candidate_rows.append(row_idx[kept])
    candidate_rows = np.concatenate(candidate_rows) if candidate_rows else np.array([], dtype=int)

    # A "good call" needs: full trajectory available, a real move (not
    # noise), the right SIGN, magnitude that isn't a token gesture (>=
    # min_magnitude_ratio of the real move -- otherwise a huge event where
    # the model predicted only a small fraction of it would still pass a
    # sign-only check and look bad on the page), AND combined must actually
    # beat Stage 1 on THIS event -- otherwise showing it proves nothing about
    # solar wind's contribution (persistence alone would have done as well
    # or better).
    magnitude_ratio = np.divide(
        np.abs(pred_change), np.abs(actual_change),
        out=np.zeros_like(actual_change), where=np.abs(actual_change) > 1e-9,
    )
    # How far combined had already committed to the call by the HALFWAY
    # horizon, relative to the eventual real move -- a call that only
    # diverges from the flat anchor (persistence) right at the last horizon
    # isn't really an onset call, it's persistence with a late correction.
    # This directly targets "good prediction of onset WITHOUT just following
    # persistence": require the combined forecast to have already moved a
    # real fraction of the way, in the right direction, well before the end.
    mid_h = max(1, out_hours // 2)
    pred_mid_change = combined_abs[:, mid_h - 1] - anchor_test
    early_divergence_ratio = np.divide(
        np.abs(pred_mid_change), np.abs(actual_change),
        out=np.zeros_like(actual_change), where=np.abs(actual_change) > 1e-9,
    )
    # A genuine "onset" story needs a quiet BEFORE too: the actual value
    # should still be near its anchor at quiet_lead_hours in (the storm
    # hasn't hit yet), AND the combined forecast should already be tracking
    # the actual closely there (small absolute error) -- so the plotted
    # panel shows "matches reality while calm, THEN correctly breaks for the
    # oncoming event" instead of already diverging (rightly or wrongly)
    # from the very first predicted hour.
    lead_h = max(1, min(quiet_lead_hours, out_hours - 1))
    actual_lead = actual_matrix[:, lead_h - 1]
    quiet_actual_ratio = np.divide(
        np.abs(actual_lead - anchor_test), np.abs(actual_change),
        out=np.full_like(actual_change, np.inf), where=np.abs(actual_change) > 1e-9,
    )
    quiet_pred_err = np.abs(actual_lead - combined_abs[:, lead_h - 1])
    filters = [
        ("has full trajectory", count == out_hours),
        ("real move (|actual| >= min_change)", np.abs(actual_change) >= min_change),
        ("combined got the sign right", np.sign(actual_change) == np.sign(pred_change)),
        ("combined magnitude not a token gesture", magnitude_ratio >= min_magnitude_ratio),
        ("combined beat Stage 1 on this event", improvement > 0),
        (f"combined already committed by h{mid_h} (not a late persistence correction)",
         (np.sign(pred_mid_change) == np.sign(actual_change)) & (early_divergence_ratio >= min_early_divergence_ratio)),
        (f"quiet before onset (actual barely moved by h{lead_h}, combined matched it there)",
         np.isfinite(quiet_actual_ratio) & (quiet_actual_ratio <= max_quiet_actual_ratio)
         & (quiet_pred_err <= max_quiet_pred_err)),
    ]
    good_mask = np.ones(n, dtype=bool)
    for label, mask in filters:
        good_mask &= mask
        survivors = int(np.isin(candidate_rows, np.flatnonzero(good_mask)).sum())
        print(f"  zoom-plot candidate filter '{label}': {survivors}/{len(candidate_rows)} sampled origins remain")
    good = candidate_rows[good_mask[candidate_rows] & np.isfinite(fan_rmse_all[candidate_rows])]
    # Onset examples (momentum was blind) ranked first; within each tier,
    # biggest improvement over Stage 1, then biggest actual move.
    good = sorted(good, key=lambda r: (not is_onset[r], -improvement[r], -abs(actual_change[r])))

    def window_for(r):
        origin_t = test_times[r]
        start = max(origin_t - pd.Timedelta(hours=margin_hours), decay.index.min())
        end = min(origin_t + out_td + pd.Timedelta(hours=margin_hours), decay.index.max())
        return start, end

    def overlaps(a, b):
        return a[0] < b[1] and b[0] < a[1]

    # ---- Select up to max_segments non-overlapping examples, trying for at
    # least one decay and one increase example before filling the rest by
    # RMSE alone.
    selected, selected_windows = [], []

    def try_add(r):
        w = window_for(r)
        if any(overlaps(w, sw) for sw in selected_windows):
            return False
        selected.append(r)
        selected_windows.append(w)
        return True

    for target_sign in (-1, 1):
        for r in good:
            if r in selected:
                continue
            if np.sign(actual_change[r]) == target_sign and try_add(r):
                break
    for r in good:
        if len(selected) >= max_segments:
            break
        if r in selected:
            continue
        try_add(r)
    selected = sorted(selected, key=lambda r: test_times[r])

    if not selected:
        print(f"Saved per-segment zoom plot (0 qualifying examples out of {len(good)} passing all filters "
              f"-- see the per-filter survivor counts above to see which condition is binding; try "
              f"lowering --zoom-min-change/--zoom-min-magnitude-ratio) to {output_path}")
        return

    fig, axes = plt.subplots(len(selected), 1, figsize=(5.2, 5.2 * len(selected)), squeeze=False)
    axes = axes[:, 0]
    fig.patch.set_facecolor("white")

    # Export the selected events' population-level forecast trajectories so
    # a downstream per-object step (e.g. individual_decay/back_out_*) can
    # consume exactly what's plotted here without re-deriving the selection.
    event_rows = []
    for event_idx, flagship in enumerate(selected):
        origin_t = test_times[flagship]
        for h in range(0, out_hours + 1):
            abs_time = origin_t if h == 0 else origin_t + pd.Timedelta(hours=h)
            event_rows.append(dict(
                event_index=event_idx, segment_id=int(test_seg_idx[flagship]),
                origin_time=origin_t, hour=h, abs_time=abs_time,
                anchor=float(anchor_test[flagship]),
                combined=float(anchor_test[flagship]) if h == 0 else float(combined_abs[flagship, h - 1]),
                combined_lo=float(anchor_test[flagship]) if h == 0 else float(combined_lo_abs[flagship, h - 1]),
                combined_hi=float(anchor_test[flagship]) if h == 0 else float(combined_hi_abs[flagship, h - 1]),
                stage1=float(anchor_test[flagship]) if h == 0 else float(stage1_abs[flagship, h - 1]),
                is_onset=bool(is_onset[flagship]),
            ))
    events_csv_path = Path(output_path).with_name(Path(output_path).stem + "_events.csv")
    pd.DataFrame(event_rows).to_csv(events_csv_path, index=False)
    print(f"Saved {len(selected)} zoom event forecast trajectories to {events_csv_path}")

    for ax, flagship in zip(axes, selected):
        plot_start, plot_end = window_for(flagship)
        window = decay.loc[plot_start:plot_end]
        seg_id = int(test_seg_idx[flagship])

        local_rows = candidate_rows[
            (test_times[candidate_rows] >= plot_start) & (test_times[candidate_rows] <= plot_end)
        ]
        n_show = min(max_fans_per_segment, len(local_rows))
        order = np.argsort(np.where(np.isnan(fan_rmse_all[local_rows]), np.inf, fan_rmse_all[local_rows]))
        shown_rows = list(local_rows[order[:n_show]])
        if flagship not in shown_rows:
            shown_rows = [flagship] + shown_rows[:max(0, n_show - 1)]

        # Numbers previously shown in an on-figure title now just print --
        # keeps the plot clean while the flagship's own full-trajectory
        # RMSE/coverage stays available. (Flagship-specific, not aggregated
        # over other nearby candidates in the window, to stay consistent
        # with the "combined beat Stage 1" selection criterion.)
        kind = "decay" if actual_change[flagship] < 0 else "increase"
        onset_label = "onset" if is_onset[flagship] else "momentum already trending"
        print(f"  zoom panel: segment {seg_id} {kind} at {test_times[flagship]:%Y-%m-%d %H:%M} "
              f"({onset_label}) -- actual {actual_change[flagship]:+.2f}, combined "
              f"{pred_change[flagship]:+.2f}, Stage 1 {stage1_change[flagship]:+.2f}, "
              f"RMSE combined={float(fan_rmse_all[flagship]):.3f} vs. Stage 1="
              f"{float(stage1_rmse_all[flagship]):.3f}, band coverage="
              f"{(in_band_count[flagship] / in_band_total[flagship] if in_band_total[flagship] else float('nan')):.0%}")

        ax.plot(window.index, window.to_numpy(), color="black", linewidth=1.8,
                marker="o", markersize=3, label="Consensus", zorder=3)

        origin_t = test_times[flagship]
        stage1_times = [origin_t] + [origin_t + pd.Timedelta(hours=h) for h in range(1, out_hours + 1)]
        stage1_values = [anchor_test[flagship]] + list(stage1_abs[flagship, :])
        ax.plot(stage1_times, stage1_values, color="#2E7D32", linestyle="--", linewidth=1.8,
                alpha=0.9, marker="o", markersize=2.8, zorder=2, label="Momentum")

        for i, r in enumerate(shown_rows):
            origin_t = test_times[r]
            fan_times = [origin_t] + [origin_t + pd.Timedelta(hours=h) for h in range(1, out_hours + 1)]
            fan_values = [anchor_test[r]] + list(combined_abs[r, :])
            fan_lo = [anchor_test[r]] + list(combined_lo_abs[r, :])
            fan_hi = [anchor_test[r]] + list(combined_hi_abs[r, :])
            is_flagship = r == flagship
            ax.fill_between(fan_times, fan_lo, fan_hi, color="tab:blue", alpha=0.22 if is_flagship else 0.12,
                             zorder=1, label="10-90% Band" if i == 0 else None)
            ax.plot(fan_times, fan_values, color="tab:blue", linewidth=2.2 if is_flagship else 1.2,
                    alpha=1.0 if is_flagship else 0.6, marker="o", markersize=3.2 if is_flagship else 2.0,
                    zorder=3, label=f"Forecast" if i == 0 else None)

        ax.set_xlim(window.index.min(), window.index.max())
        ax.set_box_aspect(1)
        ax.set_ylabel("Scaled, Normalized Decay Rate", fontsize=15)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis="both", labelsize=11)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        _style_axis(ax)
        ax.legend(loc="best", fontsize=13, framealpha=0.9, edgecolor="black")

    fig.tight_layout(h_pad=3.0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    n_onset_selected = sum(1 for r in selected if is_onset[r])
    print(f"Saved per-segment zoom plot ({len(selected)} examples where combined beat Stage 1, "
          f"{n_onset_selected} at storm onset, out of {len(good)} qualifying candidates) to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")

    p.add_argument("--indir", default=DEFAULT_INDIR)
    p.add_argument("--satcat", default=DEFAULT_SATCAT)
    p.add_argument("--dscovr-csv", default=DEFAULT_DSCOVR_CSV)
    p.add_argument("--segments-csv", default=DEFAULT_SEGMENTS_CSV,
                   help="Stormtime segments CSV from find_stormtime_segments.py")
    p.add_argument("--train-on-all-hours", action=argparse.BooleanOptionalAction, default=False,
                    help="Train/val on every chronological hour (purged by --val-date/--split-date), "
                         "not just the ~40%% of hours inside a padded stormtime segment. Test is always "
                         "restricted to the held-out storm segments regardless of this flag -- only "
                         "what's used for fitting changes. Off by default (matches the original "
                         "stormtime-only design); worth trying since two separate attempts to extract "
                         "more from the same ~9-11k stormtime-only rows (a joint model, and giving "
                         "Stage 2 momentum features) both just overfit.")
    p.add_argument("--object-type", nargs="+", default=None)
    p.add_argument("--weight-method", choices=["exponential", "inverse"], default="exponential")
    p.add_argument("--tau-minutes", type=float, default=144.0)
    p.add_argument("--min-gap-minutes", type=float, default=6.0)
    p.add_argument("--max-fill-minutes", type=int, default=15)

    p.add_argument("--input-hours", type=int, default=16,
                    help="Trailing solar-wind window to featurize (see gbm_single_horizon.py's sweep). "
                         "Not independently re-tuned for the stormtime-restricted, smaller sample regime.")
    p.add_argument("--decay-history-hours", type=int, default=16)
    p.add_argument("--n-subwindows", type=int, default=2)
    p.add_argument("--relax-ratio-features", action=argparse.BooleanOptionalAction, default=True,
                    help="Add, per channel/window, a relaxation ratio (recent max - now) / "
                         "(recent max - recent min) -- 0 = still at peak, 1 = fully relaxed back to "
                         "the window floor. Motivated by Stage 1 (momentum) being structurally blind to "
                         "storm recoveries; gives Stage 2 an explicit 'driving has relaxed' signal instead "
                         "of raw max/min levels alone. NOTE: internally uses the same literal "
                         "instantaneous minute reading as --now-feature, so it drops the same rows to NaN "
                         "during a DSCOVR gap -- turning this off alone does not recover any more usable "
                         "rows unless --no-now-feature is also set.")
    p.add_argument("--now-feature", action=argparse.BooleanOptionalAction, default=True,
                    help="Add, per channel, a literal instantaneous '_now' minute reading (not a window "
                         "aggregate). On by default -- but this flipped sign depending on --max-gate: "
                         "with the old unconstrained/loosely-capped gate (4.0), '_now' let the gate "
                         "overfit its raw scale and actively hurt (e.g. core-storm slice +16.7%% off vs. "
                         "+5.4%% on). With the gate properly capped at 0.6, it's the opposite: '_now' "
                         "carries real signal (isolated solar-wind delta-corr peaks at 0.45 vs. 0.39 "
                         "without it) and turning it on beats --no-now-feature at every horizon from h6 "
                         "on (e.g. h8 skill-vs-Stage1 +9.1%% vs. +6.9%%; core-storm +20.9%% vs. +17.6%% "
                         "vs. persistence), at a small cost at h2-h5. Also fragile to DSCOVR data gaps at "
                         "the exact origin minute even when the surrounding rolling-window stats are "
                         "still valid -- re-verify this choice if --max-gate changes materially.")
    p.add_argument("--short-window-hours", type=int, default=0)
    p.add_argument("--trend-hours", type=int, default=2)
    p.add_argument("--lag-features", choices=["both", "window", "lags", "none"], default="window")
    p.add_argument("--window-stats", choices=["minmax", "richer"], default="minmax",
                    help="Per-channel/window aggregate stats: 'minmax' is just rolling max+min (the "
                         "original encoding); 'richer' also adds rolling mean+std, so the model can tell "
                         "a channel that sat high the whole window from one that spiked then fell.")
    p.add_argument("--shock-features", action=argparse.BooleanOptionalAction, default=True,
                    help="Add, per channel, the biggest --shock-hours-scale jump (max and min signed "
                         "difference) seen ANYWHERE in the trailing --input-hours window -- an abrupt "
                         "shock/trigger signature (sudden Bz southward turn, speed/density jump at a "
                         "shock front) that the smoother --window-stats max/min/mean/std can dilute or "
                         "miss, motivated by wanting solar wind to carry a signal at storm ONSET when "
                         "Stage 1/momentum is structurally blind (see --onset-momentum-threshold).")
    p.add_argument("--shock-hours", type=int, default=1,
                    help="Timescale (hours) of the jump --shock-features looks for (value now vs. value "
                         "--shock-hours ago), before taking the max/min of that difference over the "
                         "trailing --input-hours window.")
    p.add_argument("--feature-set", choices=["full", "trimmed"], default="full")
    p.add_argument("--out-hours", type=int, default=OUT_HOURS,
                    help="Predict a shared window of horizons 1..--out-hours at once (one horizon-"
                         "conditioned model), instead of a single fixed horizon.")
    p.add_argument("--val-date", default="2025-01-01",
                    help="Storm segments starting before this go to train. Default keeps training data "
                         "spanning 2022 through late April 2025.")
    p.add_argument("--split-date", default="2025-04-01",
                    help="Storm segments starting on/after this (and before --end-date) go to test; "
                         "those in [--val-date, --split-date) go to val. Whole segments only -- see "
                         "module docstring.")
    p.add_argument("--end-date", default="2025-07-01",
                    help="Storm segments starting on/after this are dropped from EVERY split entirely "
                         "(not just pushed into test). Default 2025-07-01: the DSCOVR feed has real "
                         "outages from July 2025 onward (checked directly -- July is 53% NaN in "
                         "proton_speed, August/September are 100% NaN, Oct-Dec 36-49% NaN), so segments "
                         "in and after that stretch have almost no usable forecast origins and make the "
                         "zoom plot look sparse/broken even though the model is fine. Without this cutoff, "
                         "--split-date's 'everything after' is unbounded and lands test there by default. "
                         "Pass '' to disable and use all segments through the end of the segments CSV.")
    p.add_argument("--max-segment-hours", type=float, default=2000.0,
                    help="Segments longer than this are treated as OVERSIZED: their rows are assigned "
                         "to train/val/test individually by date instead of kept whole (see "
                         "assign_segment_buckets()'s docstring). Default 2000h (~83 days) sits between "
                         "the largest ordinary merged segment observed (~1995h) and a pathological one "
                         "(~7009h/292 days, ~59%% of all stormtime hours) caused by frequent storms "
                         "chaining together via padding during an active period -- that segment can't "
                         "be meaningfully placed in any one bucket otherwise. Pass a very large number "
                         "to disable and always keep segments whole.")

    p.add_argument("--objective", choices=["regression", "huber"], default="huber")
    p.add_argument("--huber-alpha", type=float, default=0.9)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--num-leaves", type=int, default=15)
    p.add_argument("--min-data-in-leaf", type=int, default=20)
    p.add_argument("--num-boost-round", type=int, default=2000)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--stage1-ridge-alpha", type=float, default=150.0,
                    help="Higher = simpler/weaker momentum-only Stage 1, leaving more of the target "
                         "for Stage 2 (solar wind) to explain. Default 150, re-swept for the "
                         "stormtime-only regime (60/100/150/200/300 tested): 150 gave the best "
                         "skill-vs-Stage1 in the 5-9h band specifically (e.g. +4.1%/+4.1%/+3.5%/+2.9% "
                         "at h=6/7/8/9 vs. only +3.5/+3.4/+2.6/+1.7% at the old default of 60); 300 "
                         "started to regress. Note gbm_single_horizon.py's own sweep (also 60) was done "
                         "on the full year of data, not this smaller stormtime-only sample -- the two "
                         "aren't expected to share an optimum.")
    p.add_argument("--max-gate", type=float, default=0.6,
                    help="Ceiling on the per-horizon solar-wind calibration gate(s) fit on the "
                         "validation set. Was raised to 4.0 at one point to let the asymmetric "
                         "positive-direction gate amplify Stage 2's undersized raw recovery signal, but "
                         "measured (with --no-now-feature) to badly overfit the small validation set: "
                         "unconstrained gates of 1.4-2.4 at h6-h10 lost to Stage1 by 54-75%. Swept the "
                         "cap down (4.0 -> 1.5 -> 1.0 -> 0.6 -> 0.4), and skill-vs-Stage1 improved "
                         "monotonically at every horizon all the way to 0.6 (beats Stage1 at ALL 10 "
                         "horizons there, +1.3% to +6.9%, R^2 gain from adding solar wind finally "
                         "positive at +0.059) then roughly plateaued from 0.6 to 0.4 on that metric while "
                         "continuing to erode the core-storm slice (dsn < -2 vs Stage1: +19.9% at 1.0, "
                         "+15.0% at 0.6, +10.9% at 0.4) -- 0.6 is the sweet spot, not the extreme of the "
                         "sweep. If the printed gates come back pinned at this ceiling, that's the gate "
                         "wanting to amplify further than generalizes well -- confirm with a val/test "
                         "comparison across a few values before raising this rather than trusting the "
                         "amplification.")
    p.add_argument("--architecture", choices=["two-stage", "joint"], default="two-stage",
                    help="'two-stage' (default): Ridge on decay-momentum only (Stage 1) provides an "
                         "additive floor, then LightGBM residual-on-Stage2-features (Stage 2, see "
                         "--stage2-features) corrects it, combined via a calibration gate (see "
                         "--asymmetric-gate). 'joint': ONE shared horizon-conditioned LightGBM sees "
                         "decay-momentum AND solar-wind features together and predicts the delta "
                         "directly, with no Ridge floor. TESTED WORSE: on this data, joint overfit "
                         "badly (best iteration 1303 vs. two-stage's ~40-90) and lost to persistence "
                         "by 34-91% RMSE across every horizon -- trees apparently extrapolate pure "
                         "momentum far less reliably than Ridge's smooth fit, and without the additive "
                         "floor that weakness isn't protected against. Kept available for further "
                         "experimentation (e.g. with much stronger regularization) but not recommended "
                         "as-is.")
    p.add_argument("--stage2-features", choices=["solar-wind", "all"], default="solar-wind",
                    help="--architecture two-stage only. 'solar-wind' (default): the original setup -- "
                         "Stage 2 sees only solar-wind features. 'all': also give Stage 2 the "
                         "decay-momentum features, so it can in principle learn interactions like "
                         "'momentum says decline but driving has relaxed -> predict a positive "
                         "residual.' TESTED WORSE: this let Stage 2 re-explain variance Ridge already "
                         "captured, then the asymmetric gate amplified that overfit signal (positive "
                         "gate pinned at --max-gate for 7/10 horizons; RMSE 50-87% worse than "
                         "persistence). Kept available for further experimentation but not recommended "
                         "as-is.")
    p.add_argument("--asymmetric-gate", action=argparse.BooleanOptionalAction, default=True,
                    help="--architecture two-stage only. Fit SEPARATE calibration gates for when "
                         "Stage 2's raw prediction is >= 0 (predicting recovery) vs. < 0 (predicting "
                         "decline), instead of one symmetric gate -- see "
                         "fit_solar_wind_gate_asymmetric()'s docstring. Fixes a diagnosed bias where "
                         "the model defaults to predicting continued decline even when the true outcome "
                         "is a recovery. Trade-off: costs some blanket RMSE at short horizons and raises "
                         "the false-recovery rate. On by default; pass --no-asymmetric-gate for the old "
                         "single-gate behavior.")
    p.add_argument("--event-quantile", type=float, default=0.85)
    p.add_argument("--event-weight", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--band-quantiles", type=float, nargs=2, default=[0.1, 0.9], metavar=("LOW", "HIGH"),
                    help="Two more shared horizon-conditioned LightGBM models are trained with these "
                         "quantile-regression alphas (default 10th/90th percentile, an 80%% interval) to "
                         "draw an uncertainty band around the point forecast in the zoom plot, instead of "
                         "a single line that quietly undershoots sharp storm transitions. Same Stage-1 "
                         "residual target and features as the point model; same per-horizon gate reused "
                         "to scale each quantile's contribution.")

    p.add_argument("--storm-threshold", type=float, default=-2.0,
                    help="Fixed threshold on dsn splitting the (already stormtime-only) test set into "
                         "a core-storm slice for reporting (pooled across all horizons). Default -2.0.")
    p.add_argument("--onset-momentum-threshold", type=float, default=0.15,
                    help="Test origins where Stage 1's own final-horizon |predicted change| is below "
                         "this count as 'onset' (momentum ~flat, hadn't picked up on anything yet) for "
                         "the separate onset-only metrics slice -- an honest look at whether solar wind "
                         "adds ANTICIPATORY skill, since the blanket metrics are dominated by mid-storm "
                         "rows (segments are padded around an ALREADY-detected decay crossing). Also "
                         "used by --onset-weight to upweight these rows during training.")
    p.add_argument("--onset-weight", type=float, default=1.5,
                    help="Extra training sample-weight multiplier (stacks with --event-weight) for "
                         "origins where Stage 1's own final-horizon predicted change is below "
                         "--onset-momentum-threshold -- i.e. push the GBM to specifically get better at "
                         "the rare 'momentum is blind, solar wind must carry the whole signal' case, not "
                         "just the much more common mid-storm refinement case. 1.0 disables this.")
    p.add_argument("--top-features", type=int, default=15)

    p.add_argument("--plot", default="plots/gbm_stormtime_validation.png")
    p.add_argument("--zoom-plot", default="plots/gbm_stormtime_zoom.png")
    p.add_argument("--zoom-margin-hours", type=float, default=10.0,
                    help="Hours of context to show before/after each shown example's forecast origin "
                         "(not a segment's full start/end) in the per-segment zoom plot -- a tight zoom "
                         "on the event itself.")
    p.add_argument("--zoom-fan-stride-hours", type=float, default=4.0,
                    help="Minimum spacing between candidate forecast-origin 'fans' sampled per test "
                         "segment before ranking for the zoom plot (each fan is the complete "
                         "1..--out-hours forecast trajectory launched from one origin, shown as a "
                         "shaded band -- see --band-quantiles -- not just a line).")
    p.add_argument("--zoom-max-segments", type=int, default=2,
                    help="Show up to N examples in the zoom plot -- a curated showcase of GOOD CALLS "
                         "(forecast origins where the model correctly called a real decay or a real "
                         "increase, see --zoom-min-change), ranked by LARGEST actual move first (ties "
                         "broken by lowest RMSE), not an exhaustive audit. Selection tries to include at "
                         "least one decay example and one increase example, and skips any candidate "
                         "whose zoomed window overlaps one already chosen, so shown examples are "
                         "non-overlapping distinct events.")
    p.add_argument("--zoom-min-change", type=float, default=1.0,
                    help="Minimum |actual net change| (in z-units) over the full 1..--out-hours horizon "
                         "for a forecast origin to count as a real decay/increase event worth "
                         "showcasing, rather than noise. Only origins where the combined forecast's "
                         "predicted net change agreed in sign with this real move are eligible. Raised "
                         "from 0.3: small qualifying events were easy to find but didn't read as "
                         "dramatic storms: n=4 qualified across ALL filters at 0.3 vs. only n=3 at 1.0, "
                         "but the 1.0 examples are real double-digit-percent storm-scale events (actual "
                         "changes of +2.48 and -1.45) rather than +/-0.3-0.4 wiggles. Lower this if no "
                         "qualifying examples are found.")
    p.add_argument("--zoom-min-magnitude-ratio", type=float, default=0.5,
                    help="Minimum |predicted change| / |actual change| for a sign-correct call to count "
                         "as a genuine 'good result' rather than a token gesture -- without this, ranking "
                         "by the biggest real move alone can surface events where the model got the "
                         "direction right but predicted only a small fraction of the actual size (looks "
                         "bad on the page despite technically being a 'hit'). Lower this to allow bigger "
                         "but more magnitude-undershot events back in if too few examples qualify.")
    p.add_argument("--zoom-onset-momentum-threshold", type=float, default=0.3,
                    help="An origin counts as 'storm onset' (prioritized first in the zoom plot) when "
                         "Stage 1 (momentum)'s own predicted net change over the horizon has |.| below "
                         "this -- i.e. recent decay trend was still ~flat OR badly underestimating what's "
                         "coming, so any correct call there demonstrates solar wind adding real "
                         "information rather than just refining a trend momentum had already caught. "
                         "Raised from 0.15: a case with Stage 1 predicting -0.28 against an actual -1.63 "
                         "is still momentum missing the real scale of the event, not a case where "
                         "momentum already 'has it' -- excluding that from the onset tier just because "
                         "0.28 > 0.15 was too strict for what this tier is trying to capture. Separate "
                         "from --onset-momentum-threshold, which governs the honest onset-only metrics "
                         "slice and training weight, not just this showcase.")
    p.add_argument("--zoom-min-early-divergence-ratio", type=float, default=0.3,
                    help="A qualifying example must have the combined forecast already moved, in the "
                         "right direction, at least this fraction of the eventual actual move by the "
                         "HALFWAY horizon (out-hours // 2) -- not just by the final horizon. Without "
                         "this, a 'good call' could just be persistence with a late correction tacked on "
                         "near the end, which doesn't visually read as an onset prediction. Lower this if "
                         "too few examples qualify.")
    p.add_argument("--zoom-quiet-lead-hours", type=int, default=2,
                    help="How many hours into the horizon to check for a quiet 'before the storm' period "
                         "-- see --zoom-max-quiet-actual-ratio/--zoom-max-quiet-pred-err.")
    p.add_argument("--zoom-max-quiet-actual-ratio", type=float, default=0.5,
                    help="A qualifying example must have the ACTUAL value still within this fraction of "
                         "the eventual total move, relative to anchor, at --zoom-quiet-lead-hours in -- "
                         "i.e. the storm genuinely hasn't hit yet at that point, so the panel can show a "
                         "real 'before' period. Raised from 0.3: large, sharp storms don't leave as much "
                         "clean quiet runway as small ones, so a strict cap here was screening out the "
                         "most dramatic (and most useful) examples along with --zoom-min-change. Lower "
                         "this if too few examples qualify.")
    p.add_argument("--zoom-max-quiet-pred-err", type=float, default=0.5,
                    help="A qualifying example must have combined's prediction within this absolute "
                         "|error| (z-units) of the actual value at --zoom-quiet-lead-hours in -- i.e. the "
                         "forecast is demonstrably tracking reality closely during the quiet lead-in, not "
                         "just coincidentally correct about the later onset. Raised from 0.15 for the "
                         "same reason as --zoom-max-quiet-actual-ratio. Raise further if too few examples "
                         "qualify.")
    p.add_argument("--zoom-max-fans", type=int, default=1,
                    help="Within each shown example's zoomed window, draw only its N best-fitting "
                         "(lowest per-origin RMSE) forecast fans out of the --zoom-fan-stride-hours-"
                         "sampled candidates (the flagship good-call origin is always included). "
                         "Default 1 -- just the flagship forecast, no overlapping secondary fans "
                         "cluttering the panel. RMSE/coverage in the title reflect ALL local test "
                         "origins in that window, not just the ones drawn.")
    p.add_argument("--save-model", default="models/gbm_stormtime_solar_wind_to_decay")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)


if __name__ == "__main__":
    main()
