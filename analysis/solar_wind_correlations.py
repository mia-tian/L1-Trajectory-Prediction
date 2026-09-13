"""Lagged correlations between solar wind drivers and orbit decay.

This script builds the causal, recency-weighted, standardized/normalized
decay proxy from calculate_average_decay.py (over the debris/rocket-body
TLE population in --decay-indir), then measures Pearson correlations
against solar wind parameters -- from either the OMNI2 file or the
DSCOVR-derived CSV (see --sw-source) -- for lags of 0 to --max-lag-hours
hours.

The lag is interpreted as hours from the solar wind measurement to the
observed decay rate, so lag=1 correlates solar wind at time t with decay
at time t + 1 hour.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
	standardize,
)

DEFAULT_SW_FILE = "data/omni2_EmUtmxgIi3.lst.txt"
DEFAULT_DSCOVR_CSV = "dscovr_2024/dscovr_2024_core_and_derived.csv"
DEFAULT_OUTPUT_CSV = "plots/solar_wind_decay_lag_correlations.csv"
DEFAULT_OUTPUT_PNG = "plots/solar_wind_decay_lag_correlations.png"
DEFAULT_DECAY_PNG = "plots/averaged_standardized_decay_proxy.png"

# ---------------------------------------------------------------------------
# Toggle which parameters go into the correlation comparison HERE -- flip a
# value to False and re-run, no CLI flags needed. Keyed by raw column name
# (as it appears in the DSCOVR CSV / OMNI2 loader), not the display name used
# in the plot -- "bz_gsm" shows up as "bz_south" in the output (it's
# sign-flipped, see lagged_correlations()), so toggle it by "bz_gsm" here.
# Any column not listed defaults to included (True) -- this is a toggle list,
# not an allowlist, so new derived columns show up automatically.
# ---------------------------------------------------------------------------
PARAM_TOGGLES: dict[str, bool] = {
    "proton_speed": False,
    "proton_density": False,
    "proton_temperature": False,
    "bt": True,
    "by_gsm": False,
    "bz_gsm": False,  # displayed as "bz_south"
    "bs": True,
    "bt_yz": False,
    "theta_gsm": False,
    "clock_angle_deg": False,
    "dynamic_pressure_npa": False,
    "ey_southward_mvm": True,
    "kan_lee_mvm": True,
    "newell_coupling": True,
    "akasofu_epsilon_gw": True,
    "speed_change_kms_per_min": False,
    "pressure_change_npa_per_min": False,
    "southward_fraction_60m": False,
    "bz_min_60m": False,
    "ey_mean_60m": False,
    "ey_integral_1h": False,
    "ey_integral_3h": False,
    "ring_current_proxy_tau3h": True,
    "ring_current_proxy_tau8h": True,
    "ring_current_proxy_obrien": False,
    "kp_index": False,
    "dst_index": False,
}



# Raw column/parameter name -> polished display name, used only for the
# plot_lag_lines() legend (keyed post bz_gsm->bz_south rename, since that's
# the name the plot actually sees). Anything not listed here falls back to
# its raw name unchanged, so new derived columns still show up.
DISPLAY_NAMES: dict[str, str] = {
    "proton_speed": "Proton Speed",
    "proton_density": "Proton Density",
    "proton_temperature": "Proton Temperature",
    "bt": "IMF Magnitude (Bt)",
    "by_gsm": "IMF By (GSM)",
    "bz_south": "IMF Bz South (GSM)",
    "bz_gsm": "IMF Bz (GSM)",  # population_proxy/*.py's SW_FEATURES use "bz_gsm" directly (no bz_south rename)
    "bs": "Southward B (Bs)",
    "bt_yz": "IMF Magnitude, Y-Z Plane (Bt,yz)",
    "theta_gsm": "IMF Clock Angle (GSM)",
    "clock_angle_deg": "IMF Clock Angle (deg)",
    "dynamic_pressure_npa": "Dynamic Pressure (nPa)",
    "ey_southward_mvm": "Southward Electric Field Ey (mV/m)",
    "kan_lee_mvm": "Kan-Lee Coupling Function (mV/m)",
    "newell_coupling": "Newell Coupling Function",
    "akasofu_epsilon_gw": "Akasofu Epsilon Parameter (GW)",
    "speed_change_kms_per_min": "Speed Change (km/s per min)",
    "pressure_change_npa_per_min": "Pressure Change (nPa per min)",
    "southward_fraction_60m": "Southward Fraction (60 min)",
    "bz_min_60m": "Min Bz (60 min)",
    "ey_mean_60m": "Mean Ey (60 min)",
    "ey_integral_1h": "Ey Integral (1h)",
    "ey_integral_3h": "Ey Integral (3h)",
    "ring_current_proxy_tau3h": "Ring Current Proxy (tau=3h)",
    "ring_current_proxy_tau8h": "Ring Current Proxy (tau=8h)",
    "ring_current_proxy_obrien": "Ring Current Proxy (O'Brien)",
    "kp_index": "Kp Index",
    "dst_index": "Dst Index",
}


def apply_param_toggles(columns: list[str]) -> list[str]:
    kept = [c for c in columns if PARAM_TOGGLES.get(c, True)]
    if not kept:
        raise ValueError("PARAM_TOGGLES excludes every available parameter -- nothing to compute")
    return kept


# Suffix shape -> human-readable description, tried in order. Covers every
# engineered-feature naming convention used across population_proxy/*.py's
# build_features()/build_lag_features(): window max/min (with or without
# sub-window chunking), literal hourly lags (gbm/xgb use "_lag_Nh", dlm uses
# "_lagNh" -- the optional underscore below covers both), and trend deltas.
_FEATURE_SUFFIX_PATTERNS = [
    (re.compile(r"^_max_(\d+)h_chunk(\d+)$"),
     lambda m: f"max, {m.group(1)}h window (part {int(m.group(2)) + 1})"),
    (re.compile(r"^_min_(\d+)h_chunk(\d+)$"),
     lambda m: f"min, {m.group(1)}h window (part {int(m.group(2)) + 1})"),
    (re.compile(r"^_max_(\d+)h$"), lambda m: f"max, {m.group(1)}h window"),
    (re.compile(r"^_min_(\d+)h$"), lambda m: f"min, {m.group(1)}h window"),
    (re.compile(r"^_mean_(\d+)h_chunk(\d+)$"),
     lambda m: f"mean, {m.group(1)}h window (part {int(m.group(2)) + 1})"),
    (re.compile(r"^_std_(\d+)h_chunk(\d+)$"),
     lambda m: f"std, {m.group(1)}h window (part {int(m.group(2)) + 1})"),
    (re.compile(r"^_mean_(\d+)h$"), lambda m: f"mean, {m.group(1)}h window"),
    (re.compile(r"^_std_(\d+)h$"), lambda m: f"std, {m.group(1)}h window"),
    (re.compile(r"^_trend_(\d+)h$"), lambda m: f"trend ({m.group(1)}h)"),
    (re.compile(r"^_shock(\d+)h_max_(\d+)h$"),
     lambda m: f"biggest {m.group(1)}h jump up, {m.group(2)}h window"),
    (re.compile(r"^_shock(\d+)h_min_(\d+)h$"),
     lambda m: f"biggest {m.group(1)}h jump down, {m.group(2)}h window"),
    (re.compile(r"^_lag_?(\d+)h$"), lambda m: f"lag {m.group(1)}h"),
    (re.compile(r"^_relax_(\d+)h_chunk(\d+)$"),
     lambda m: f"relaxation from peak, {m.group(1)}h window (part {int(m.group(2)) + 1})"),
    (re.compile(r"^_relax_(\d+)h$"), lambda m: f"relaxation from peak, {m.group(1)}h window"),
    (re.compile(r"^_now$"), lambda m: "now"),
]


def prettify_engineered_feature_name(raw: str, channel_names: dict[str, str] = DISPLAY_NAMES) -> str:
    """Polish an engineered feature name (e.g. from gbm/xgb/dlm_single_horizon.py's
    build_features()/build_lag_features()) for plot/print display, e.g.
    "ring_current_proxy_tau3h_min_6h_chunk0" -> "Ring Current Proxy (tau=3h)
    -- min, 6h window (part 1)". Falls back to the raw name wherever the
    channel or suffix shape isn't recognized, rather than guessing wrong."""
    if raw == "decay_level":
        return "Decay Level (momentum)"
    decay_change = re.match(r"^decay_change_(\d+)h$", raw)
    if decay_change:
        return f"Decay Change ({decay_change.group(1)}h)"

    # Longest-prefix match against known channel names -- channel names can
    # themselves contain underscores (e.g. "ring_current_proxy_tau3h"), so a
    # naive split on the first "_" would cut mid-channel-name.
    channel = None
    for candidate in sorted(channel_names, key=len, reverse=True):
        if raw == candidate or raw.startswith(candidate + "_"):
            channel = candidate
            break
    if channel is None:
        return raw

    suffix = raw[len(channel):]
    polished = channel_names[channel]
    if suffix == "":
        return polished
    for pattern, fmt in _FEATURE_SUFFIX_PATTERNS:
        match = pattern.match(suffix)
        if match:
            return f"{polished} — {fmt(match)}"
    return f"{polished}{suffix}"  # unrecognized suffix shape -- keep the raw tail rather than drop information


def _clean_columns(columns: Iterable[str]) -> list[str]:
	return [column.replace("\ufeff", "").strip().strip('"') for column in columns]


def build_weighted_decay_proxy(
	indir: Path,
	satcat_path: Path,
	object_types: list[str] | None,
	weight_method: str,
	tau_minutes: float,
	min_gap_minutes: float,
	step_minutes: float,
	year: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
	"""Build the causal recency-weighted decay proxy (see calculate_average_decay.py)."""
	paths = sorted(indir.glob("*.txt"))
	if not paths:
		raise FileNotFoundError(f"No TLE files found in {indir}")

	if object_types:
		wanted = {t.upper() for t in object_types}
		satcat_types = load_satcat_types(str(satcat_path))
		paths = [p for p in paths if satcat_types.get(p.stem, "").upper() in wanted]
		if not paths:
			raise ValueError(f"No objects in {indir} match --decay-object-type {sorted(wanted)}")

	all_records = []
	for path in paths:
		pts = load_object_series(str(path))
		if len(pts) < 3:
			continue
		recs = compute_normalized_decay_rate(pts)
		if not recs:
			continue
		recs = standardize(recs)
		if recs:
			all_records.extend(recs)

	if not all_records:
		raise ValueError(f"No decay-rate measurements computed from {indir}")

	eval_times, weighted_avg, _counts = causal_weighted_average(
		all_records,
		step_minutes=step_minutes,
		method=weight_method,
		tau_minutes=tau_minutes,
		min_gap_minutes=min_gap_minutes,
	)
	decay_index = pd.DatetimeIndex(eval_times)
	return decay_index, np.asarray(weighted_avg, dtype=float)


def load_solar_wind(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Missing OMNI2 file: {path}")

	columns = [
		"year",
		"day",
		"hour",
		"minute",
		"second",
		"imf_mag_avg",
		"imf_mag",
		"by_gsm",
		"bz_gsm",
		"proton_density",
		"flow_speed",
		"flow_pressure",
		"kp",
		"sunspot",
		"dst",
		"f107",
	]

	frame = pd.read_csv(path, sep=r"\s+", header=None, names=columns)
	for column in columns:
		frame[column] = pd.to_numeric(frame[column], errors="coerce")

	frame = frame.dropna(subset=["year", "day", "hour"]).copy()
	frame["year"] = frame["year"].astype(int)
	frame["day"] = frame["day"].astype(int)
	frame["hour"] = frame["hour"].astype(int)
	frame["minute"] = frame["minute"].astype(int)
	frame["second"] = frame["second"].astype(int)

	base = pd.to_datetime(frame["year"].astype(str), format="%Y", utc=False)
	frame["DATE"] = (
		base
		+ pd.to_timedelta(frame["day"] - 1, unit="D")
		+ pd.to_timedelta(frame["hour"], unit="h")
		+ pd.to_timedelta(frame["minute"], unit="m")
		+ pd.to_timedelta(frame["second"], unit="s")
	)
	frame = frame.sort_values("DATE").reset_index(drop=True)
	frame = frame.set_index("DATE")
	return frame


def load_omni_indices(path: Path) -> pd.DataFrame:
    """Kp and Dst -- the measured geomagnetic response, not L1 solar wind --
    from the same OMNI2 file used by --sw-source omni, so they can be merged
    alongside the DSCOVR-derived columns instead of being an either/or source
    choice. Column mapping verified against the real 2024-05-11 Gannon storm
    (dst bottoms at -406 nT, kp peaks at 90 i.e. Kp=9.0, both matching the
    recorded extremes almost exactly).

    The file's "minute"/"second" fields aren't real sub-hour timestamps (this
    is hourly-averaged data) -- they hold some other constant-ish metadata --
    so the index is floored to the hour here rather than trusted as-is.
    """
    frame = load_solar_wind(path)
    frame = frame.copy()
    frame.index = frame.index.floor("h")
    frame = frame[~frame.index.duplicated(keep="first")]
    out = frame[["kp", "dst"]].rename(columns={"kp": "kp_index", "dst": "dst_index"})
    out["kp_index"] = out["kp_index"] / 10.0  # OMNI2 convention: Kp*10 as an integer
    return out


def load_dscovr(path: Path) -> pd.DataFrame:
	"""Load the year-long DSCOVR core+derived CSV built by process_dscovr_year.py."""
	if not path.exists():
		raise FileNotFoundError(f"Missing DSCOVR CSV: {path}")

	frame = pd.read_csv(path, parse_dates=["time"])
	frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None)
	frame = frame.sort_values("time").set_index("time")
	return frame


def select_solar_wind_columns(frame: pd.DataFrame) -> list[str]:
	excluded = {
		"year", "day", "hour", "minute", "second",
		# DSCOVR quality/status flags -- not physical drivers, exclude from correlation.
		"quality_fc", "fill_flag_fc", "calibration_mode_flag_fc", "maneuver_flag_fc",
		"quality_mag", "fill_flag_mag", "possible_saturation_flag",
		"calibration_mode_flag_mag", "maneuver_flag_mag", "quality_combined",
		# quantities:
		"alpha_density", "alpha_speed"
		# "bt_yz", "bz_min_60min", "clock_angle_deg", "pressure_change_npa_per_min",
		# "speed_change_kms_per_min"
	}
	return [column for column in frame.columns if column not in excluded]


def hourly_solar_wind_series(frame: pd.DataFrame, column: str, grid: pd.DatetimeIndex) -> np.ndarray:
	hourly = frame[[column]].resample("h").mean()
	hourly = hourly.reindex(grid)
	if hourly[column].isna().any():
		hourly[column] = hourly[column].interpolate(limit_direction="both")
		if hourly[column].isna().any():
			hourly[column] = hourly[column].ffill().bfill()
	values = hourly[column].to_numpy(dtype=float)
	if column == "bz_gsm":
		return np.maximum(-values, 0.0)
	return values


def lagged_correlations(
	solar_wind: pd.DataFrame,
	decay_index: pd.DatetimeIndex,
	decay_proxy: np.ndarray,
	columns: list[str],
	max_lag_hours: int = 16,
) -> pd.DataFrame:
	results = []

	for column in columns:
		sw_series = hourly_solar_wind_series(solar_wind, column, decay_index)
		parameter_name = "bz_south" if column == "bz_gsm" else column
		for lag in range(0, max_lag_hours + 1):
			if lag == 0:
				x = sw_series
				y = decay_proxy
			else:
				x = sw_series[:-lag]
				y = decay_proxy[lag:]
			mask = np.isfinite(x) & np.isfinite(y)
			corr = np.nan
			if mask.sum() >= 3:
				corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
			results.append({"parameter": parameter_name, "lag_hours": lag, "correlation": corr})

	return pd.DataFrame(results)


def plot_lag_lines(result_frame: pd.DataFrame, output_path: Path, max_hours: int = 16) -> None:
	"""One line per parameter showing Pearson correlation vs. lag, out to max_hours."""
	subset = result_frame[result_frame["lag_hours"] <= max_hours]
	pivot = subset.pivot(index="lag_hours", columns="parameter", values="correlation").sort_index()

	# Order by peak |correlation| (strongest, most relevant features first) --
	# both the legend and the color assignment follow this order, so the most
	# distinct palette colors land on the parameters actually worth telling
	# apart, and the legend reads top-to-bottom as "most to least relevant."
	ordered_params = pivot.abs().max(axis=0).sort_values(ascending=False).index.tolist()
	n = len(ordered_params)

	# tab20/tab20b/tab20c store colors as (dark, light) pairs of the same hue
	# back-to-back, so taking them in order puts near-identical hues next to
	# each other. Reorder to take all the saturated "dark" entries first (this
	# is exactly the 10-hue tab10 palette) before falling back to their
	# lighter pastel partners, so the first ~10-30 colors assigned are as
	# hue-separated as possible.
	def _reorder(cmap_colors: list) -> list:
		return list(cmap_colors[0::2]) + list(cmap_colors[1::2])

	palette = _reorder(plt.get_cmap("tab20").colors)
	if n > 20:
		palette += _reorder(plt.get_cmap("tab20b").colors)
		palette += _reorder(plt.get_cmap("tab20c").colors)
	colors = [palette[i % len(palette)] for i in range(n)]

	fig, ax = plt.subplots(figsize=(10, 5))
	for parameter, color in zip(ordered_params, colors):
		ax.plot(
			pivot.index, pivot[parameter],
			linewidth=1.2,
			label=DISPLAY_NAMES.get(parameter, parameter), color=color,
		)

	ax.axhline(0.0, color="0.4", linewidth=0.9, linestyle="--", zorder=1)
	ax.set_xlim(0, max_hours)
	ax.set_ylim(-.65, -.25)
	ax.set_xticks(range(0, max_hours + 1))
	ax.set_xlabel("Lag From Solar Wind Measurement to Decay (Hours)", fontsize=14, labelpad=8)
	ax.set_ylabel("Pearson Correlation Coefficient", fontsize=14, labelpad=8)
	# ax.set_title(f"Lagged Correlation with Weighted Decay Proxy (0–{max_hours}h)",
	# 	fontsize=15, fontweight="semibold", pad=14)
	ax.tick_params(labelsize=11)
	for spine in ("top", "right"):
		ax.spines[spine].set_visible(False)
	ax.spines["left"].set_color("0.3")
	ax.spines["bottom"].set_color("0.3")
	ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.6)
	ax.set_facecolor("#fbfbfb")
	legend = ax.legend(
		loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=11,
		frameon=False, title="Parameter (Strongest → Weakest)", title_fontsize=12,
		handlelength=1.6, labelspacing=0.6,
	)
	legend.get_title().set_ha("left")
	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def plot_average_decay(
	decay_index: pd.DatetimeIndex,
	decay_proxy: np.ndarray,
	output_path: Path,
) -> None:
	fig, ax = plt.subplots(figsize=(12, 4))
	ax.plot(decay_index, decay_proxy, color="tab:blue", linewidth=1.0, label="Weighted average")
	ax.axhline(0.0, color="tab:gray", linestyle=":", linewidth=1.0, alpha=0.7)
	ax.set_xlabel("Time")
	ax.set_ylabel("Weighted avg. standardized, normalized decay rate")
	ax.set_title("Causal recency-weighted decay proxy")
	ax.legend(loc="upper right")
	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)


def print_summary(result_frame: pd.DataFrame) -> None:
	print("Lagged correlations against the averaged standardized decay proxy:\n")
	for parameter, group in result_frame.groupby("parameter"):
		valid = group.dropna(subset=["correlation"])
		if valid.empty:
			print(f"{parameter}: no valid correlations")
			continue
		best = valid.iloc[valid["correlation"].abs().argmax()]
		print(
			f"{parameter}: best lag {int(best['lag_hours'])} h, "
			f"correlation {best['correlation']:.3f}"
		)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Correlate solar wind parameters with the causal, recency-weighted decay proxy."
	)
	# Decay proxy (see calculate_average_decay.py for what these control).
	parser.add_argument("--decay-indir", default="deb_tles_2024",
		help="Directory of per-NORAD TLE files used to build the weighted decay proxy")
	parser.add_argument("--decay-object-type", nargs="+", default=None,
		help="Restrict decay proxy to satcat OBJECT_TYPE code(s), e.g. --decay-object-type DEB")
	parser.add_argument("--decay-satcat", default="satcat.csv",
		help="Path to satcat CSV, used for --decay-object-type filtering")
	parser.add_argument("--decay-weight-method", choices=["exponential", "inverse"], default="exponential",
		help="Recency weighting scheme for the decay proxy (default 'exponential')")
	parser.add_argument("--decay-tau-minutes", type=float, default=60.0,
		help="Decay time-constant in minutes for --decay-weight-method exponential (default 60)")
	parser.add_argument("--decay-min-gap-minutes", type=float, default=6.0,
		help="Floor on (evaluation time - measurement time) in minutes, for --decay-weight-method inverse (default 6)")
	parser.add_argument("--decay-step-minutes", type=float, default=60.0,
		help="Decay-proxy evaluation grid spacing in minutes (default 60, i.e. hourly, to match lag-correlation granularity)")
	parser.add_argument("--decay-year", type=int, default=2024)

	# Solar wind source: OMNI2 file (default, original behavior) or the DSCOVR-derived CSV.
	parser.add_argument("--sw-source", choices=["omni", "dscovr"], default="dscovr",
		help="Solar wind data source: 'omni' (OMNI2 hourly file, default) or 'dscovr' "
		     "(the DSCOVR-derived CSV from process_dscovr_year.py)")
	parser.add_argument("--sw-file", default=DEFAULT_SW_FILE, help="OMNI2 file, used when --sw-source omni")
	parser.add_argument("--dscovr-csv", default=DEFAULT_DSCOVR_CSV, help="DSCOVR CSV, used when --sw-source dscovr")
	parser.add_argument("--dscovr-columns", nargs="+", default=None,
		help="Restrict calculation/plotting to these DSCOVR column(s) only, e.g. "
		     "--dscovr-columns bz_gsm bt proton_speed. Only applies when --sw-source dscovr; "
		     "default is every non-flag column in the CSV.")

	parser.add_argument("--kp-dst-file", default=DEFAULT_SW_FILE,
		help="OMNI2 file to pull Kp/Dst (measured geomagnetic response) from, merged alongside "
		     "whichever --sw-source is selected, unless --skip-kp-dst is passed")
	parser.add_argument("--skip-kp-dst", action="store_true",
		help="Don't merge in Kp/Dst from --kp-dst-file (skips loading the OMNI2 file entirely)")

	parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
	parser.add_argument("--output-png", default=DEFAULT_OUTPUT_PNG)
	parser.add_argument("--decay-png", default=DEFAULT_DECAY_PNG)
	parser.add_argument("--max-lag-hours", type=int, default=16,
		help="How many lag hours to compute correlations for (default 16)")
	parser.add_argument("--plot-max-lag-hours", type=int, default=16,
		help="How many lag hours to show on the line plot's x-axis (default 16)")
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

	if args.sw_source == "dscovr":
		solar_wind = load_dscovr(root / args.dscovr_csv)
	else:
		solar_wind = load_solar_wind(root / args.sw_file)

	columns = select_solar_wind_columns(solar_wind)
	if not columns:
		raise ValueError(f"No usable driver columns were found in the {args.sw_source} data")

	if args.sw_source == "dscovr" and args.dscovr_columns:
		requested = args.dscovr_columns
		missing = [c for c in requested if c not in columns]
		if missing:
			raise ValueError(
				f"Unknown --dscovr-columns {missing}. Available columns: {columns}"
			)
		columns = requested

	if not args.skip_kp_dst:
		kp_dst = load_omni_indices(root / args.kp_dst_file)
		solar_wind = pd.concat([solar_wind, kp_dst], axis=1)
		columns = columns + ["kp_index", "dst_index"]
		print(f"Merged in kp_index/dst_index from {args.kp_dst_file}")

	columns = apply_param_toggles(columns)
	print(f"Using {len(columns)} parameter(s) (edit PARAM_TOGGLES at the top of this file to "
	      f"change): {columns}")

	result_frame = lagged_correlations(
		solar_wind=solar_wind,
		decay_index=decay_index,
		decay_proxy=decay_proxy,
		columns=columns,
		max_lag_hours=args.max_lag_hours,
	)

	result_frame = result_frame.sort_values(["parameter", "lag_hours"]).reset_index(drop=True)

	output_csv = root / args.output_csv
	output_png = root / args.output_png
	decay_png = root / args.decay_png
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	result_frame.to_csv(output_csv, index=False)
	pd.Series(decay_proxy, index=decay_index, name="weighted_avg_decay").to_csv(
		output_csv.with_name("averaged_standardized_decay_proxy.csv"), index_label="time"
	)
	plot_average_decay(decay_index, decay_proxy, decay_png)
	plot_lag_lines(result_frame, output_png, max_hours=args.plot_max_lag_hours)

	print_summary(result_frame)
	print(f"\nSolar wind source: {args.sw_source}")
	print(f"Saved correlations to {output_csv}")
	print(f"Saved averaged decay plot to {decay_png}")
	print(f"Saved lag-correlation plot to {output_png}")


if __name__ == "__main__":
	main()
