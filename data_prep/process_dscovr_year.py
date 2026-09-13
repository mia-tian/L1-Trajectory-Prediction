"""Build one continuous CSV of DSCOVR core and derived solar-wind/IMF
quantities, spanning one year, several years, or every year found.

For each day with both a Faraday Cup (oe_f1m_*) and Magnetometer (oe_m1m_*)
file in --dscovr-dir, loads and merges the two Level-2 one-minute products,
concatenates the raw per-day merges into one continuous time series, and
computes the derived quantities ONCE on the full series -- so rate-of-change
and rolling-window (1 h / 3 h) features stay continuous across day (and year)
boundaries instead of resetting.

Example
-------
python process_dscovr_year.py --dscovr-dir dscovr --year 2024
python process_dscovr_year.py --dscovr-dir dscovr --year 2022 2023 2024
python process_dscovr_year.py --dscovr-dir dscovr              # every year found
"""

from __future__ import annotations

import argparse
import gzip
import re
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

FILENAME_DATE_RE = re.compile(r"_s(\d{8})\d{6}_")


# ---------------------------------------------------------------------------
# Per-day loading and merging (Faraday Cup + Magnetometer)
# ---------------------------------------------------------------------------

def open_netcdf(path: Path) -> xr.Dataset:
    """Open NetCDF classic data from an uncompressed or gzip-wrapped file."""
    if path.suffix != ".gz":
        return xr.open_dataset(path, decode_times=True)

    with gzip.open(path, "rb") as source, tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        tmp.write(source.read())
        tmp.flush()
        return xr.open_dataset(tmp.name, decode_times=True).load()


def clean_dataset(
    ds: xr.Dataset,
    keep: Iterable[str],
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
) -> pd.DataFrame:
    """Select variables, decode time, replace fill values and restrict to [date_start, date_end)."""
    absent = [name for name in keep if name not in ds.variables]
    if absent:
        raise KeyError(f"Required variables missing from file: {absent}")

    frame = ds[list(keep)].to_dataframe().reset_index()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.set_index("time").sort_index()

    numeric = frame.select_dtypes(include=[np.number]).columns
    frame[numeric] = frame[numeric].mask(frame[numeric] <= -99990)
    frame = frame.loc[(frame.index >= date_start) & (frame.index < date_end)]
    return frame


def load_inputs(
    faraday_path: Path,
    mag_path: Path,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
) -> pd.DataFrame:
    fc_vars = [
        "proton_speed",
        "proton_density",
        "proton_temperature",
        "alpha_speed",
        "alpha_density",
        "overall_quality",
        "fill_flag",
        "calibration_mode_flag",
        "maneuver_flag",
    ]
    mag_vars = [
        "bt",
        "by_gsm",
        "bz_gsm",
        "theta_gsm",
        "overall_quality",
        "fill_flag",
        "possible_saturation_flag",
        "calibration_mode_flag",
        "maneuver_flag",
    ]

    with open_netcdf(faraday_path) as fc_ds:
        fc = clean_dataset(fc_ds, fc_vars, date_start=date_start, date_end=date_end)
    with open_netcdf(mag_path) as mag_ds:
        mag = clean_dataset(mag_ds, mag_vars, date_start=date_start, date_end=date_end)

    fc = fc.rename(columns={
        "overall_quality": "quality_fc",
        "fill_flag": "fill_flag_fc",
        "calibration_mode_flag": "calibration_mode_flag_fc",
        "maneuver_flag": "maneuver_flag_fc",
    })
    mag = mag.rename(columns={
        "overall_quality": "quality_mag",
        "fill_flag": "fill_flag_mag",
        "calibration_mode_flag": "calibration_mode_flag_mag",
        "maneuver_flag": "maneuver_flag_mag",
    })

    # Both products are nominally one-minute averages. Nearest-time matching is
    # tolerant to small timestamp offsets while avoiding broad interpolation.
    merged = pd.merge_asof(
        fc.sort_index().reset_index(),
        mag.sort_index().reset_index(),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("40s"),
    ).set_index("time")

    merged["quality_combined"] = merged[["quality_fc", "quality_mag"]].max(axis=1)

    bad_status = (
        merged[[
            "fill_flag_fc", "fill_flag_mag",
            "calibration_mode_flag_fc", "calibration_mode_flag_mag",
            "maneuver_flag_fc", "maneuver_flag_mag",
            "possible_saturation_flag",
        ]]
        .fillna(0)
        .max(axis=1)
        .gt(0)
    )
    merged.loc[bad_status, "quality_combined"] = 2
    return merged


# ---------------------------------------------------------------------------
# Derived solar-wind/magnetosphere-coupling quantities
# ---------------------------------------------------------------------------

def add_derived_quantities(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    v = out["proton_speed"]                  # km/s, positive magnitude
    np_cm3 = out["proton_density"]           # cm^-3
    na_cm3 = out["alpha_density"]            # cm^-3
    by = out["by_gsm"]                       # nT
    bz = out["bz_gsm"]                       # nT

    out["bs"] = (-bz).clip(lower=0)          # southward field magnitude, nT
    out["bt_yz"] = np.hypot(by, bz)          # transverse IMF magnitude, nT

    # Clock angle in the GSM Y-Z plane: 0 degrees = northward, 180 = southward.
    out["clock_angle_deg"] = np.degrees(np.arctan2(by, bz)) % 360.0
    half_angle = np.deg2rad(out["clock_angle_deg"] / 2.0)

    # Dynamic pressure in nPa. Alpha particles contribute four proton masses each.
    # The alpha term uses alpha speed separately when available.
    proton_pressure = 1.6726219e-6 * np_cm3 * v.pow(2)
    alpha_pressure = 1.6726219e-6 * 4.0 * na_cm3 * out["alpha_speed"].pow(2)
    out["dynamic_pressure_npa"] = proton_pressure + alpha_pressure.fillna(0.0)

    # Geoeffective dawn-dusk electric field magnitude, mV/m.
    # 1e-3 converts (km/s * nT) to mV/m.
    out["ey_southward_mvm"] = 1.0e-3 * v * out["bs"]

    # Kan-Lee coupling electric field, mV/m.
    out["kan_lee_mvm"] = 1.0e-3 * v * out["bt_yz"] * np.sin(half_angle).pow(2)

    # Newell reconnection coupling proxy. Conventional relative units.
    out["newell_coupling"] = (
        v.clip(lower=0).pow(4.0 / 3.0)
        * out["bt_yz"].clip(lower=0).pow(2.0 / 3.0)
        * np.abs(np.sin(half_angle)).pow(8.0 / 3.0)
    )

    # Akasofu epsilon coupling proxy in GW:
    # epsilon = (4*pi/mu0) * V * B^2 * sin^4(theta/2) * l0^2, l0 = 7 RE.
    mu0 = 4.0 * np.pi * 1.0e-7
    earth_radius_m = 6_371_000.0
    l0 = 7.0 * earth_radius_m
    out["akasofu_epsilon_gw"] = (
        (4.0 * np.pi / mu0)
        * (v * 1_000.0)
        * (out["bt_yz"] * 1.0e-9).pow(2)
        * np.sin(half_angle).pow(4)
        * l0**2
        / 1.0e9
    )

    # One-minute rates of change. Time-based division remains valid if samples are missing.
    dt_min = out.index.to_series().diff().dt.total_seconds().div(60.0)
    out["speed_change_kms_per_min"] = v.diff().div(dt_min)
    out["pressure_change_npa_per_min"] = out["dynamic_pressure_npa"].diff().div(dt_min)

    # Historical/accumulated forcing features useful for density prediction.
    out["southward_fraction_60m"] = bz.lt(0).astype(float).rolling("60min", min_periods=10).mean()
    out["bz_min_60m"] = bz.rolling("60min", min_periods=10).min()
    out["ey_mean_60m"] = out["ey_southward_mvm"].rolling("60min", min_periods=10).mean()

    # Time integral of E_y over trailing 1 h and 3 h. Units: mV/m * min.
    out["ey_integral_1h"] = out["ey_southward_mvm"].rolling("60min", min_periods=10).sum()
    out["ey_integral_3h"] = out["ey_southward_mvm"].rolling("180min", min_periods=30).sum()

    # Synthetic ring-current-like injection filter (Burton 1975 / O'Brien &
    # McPherron 2000 form): dQ*/dt = a*max(VBs - VBs_c, 0) - Q*/tau, driven
    # purely by solar wind (VBs = ey_southward_mvm above) -- no measured Dst
    # is used anywhere. This gives a causal state variable with genuine
    # multi-hour charge/discharge memory (the mechanism behind the ~6-10h
    # solar-wind/decay lag seen empirically), instead of the ad hoc rolling
    # max/min windows used elsewhere. Two fixed time constants bracket
    # substorm (~3h) vs. classic ring-current (~7.7h) scales; the third uses
    # O'Brien & McPherron's driving-dependent tau (faster decay under
    # stronger driving).
    dt_hr = dt_min.to_numpy() / 60.0
    dt_hr = np.where(np.isfinite(dt_hr) & (dt_hr > 0), dt_hr, 1.0 / 60.0)
    dt_hr = np.clip(dt_hr, 0.0, 3.0)  # cap large data gaps so state can't blow up on resume
    vbs_clean = np.nan_to_num(out["ey_southward_mvm"].to_numpy(), nan=0.0)

    def _ring_current_filter(injection_a: float, vbs_threshold: float, tau_hours) -> np.ndarray:
        n = len(vbs_clean)
        q = np.empty(n)
        variable_tau = callable(tau_hours)
        state = 0.0
        for i in range(n):
            driving = injection_a * max(vbs_clean[i] - vbs_threshold, 0.0)
            tau_i = tau_hours(vbs_clean[i]) if variable_tau else tau_hours
            state = state * np.exp(-dt_hr[i] / tau_i) + driving * dt_hr[i]
            q[i] = state
        return q

    out["ring_current_proxy_tau3h"] = _ring_current_filter(4.4, 0.5, 3.0)
    out["ring_current_proxy_tau8h"] = _ring_current_filter(4.4, 0.5, 7.7)
    out["ring_current_proxy_obrien"] = _ring_current_filter(
        4.4, 0.5, lambda v: 2.40 * np.exp(9.74 / (4.69 + max(v, 0.0)))
    )

    return out


# ---------------------------------------------------------------------------
# Multi-day / multi-year assembly
# ---------------------------------------------------------------------------

def discover_day_files(dscovr_dir: Path, prefix: str) -> dict[str, Path]:
    """Map YYYYMMDD -> file path for all oe_<prefix>_dscovr_s*_pub.nc(.gz) files."""
    days: dict[str, Path] = {}
    for path in sorted(dscovr_dir.glob(f"oe_{prefix}_dscovr_s*_pub.nc*")):
        match = FILENAME_DATE_RE.search(path.name)
        if not match:
            continue
        days[match.group(1)] = path
    return days


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dscovr-dir", default="dscovr", help="Directory of per-day DSCOVR files")
    parser.add_argument(
        "--year", type=int, nargs="+", default=None,
        help="Calendar year(s) to include, e.g. --year 2024 or --year 2022 2023 2024. "
             "Default: every year with data found in --dscovr-dir.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output CSV path. Default: dscovr_<year>/dscovr_<year>_core_and_derived.csv for a "
             "single year, or dscovr_<minyear>_<maxyear>/dscovr_<minyear>_<maxyear>_core_and_derived.csv "
             "for multiple years.",
    )
    args = parser.parse_args()

    dscovr_dir = Path(args.dscovr_dir).expanduser()
    fc_files = discover_day_files(dscovr_dir, "f1m")
    mag_files = discover_day_files(dscovr_dir, "m1m")

    available_days = fc_files.keys() & mag_files.keys()
    if args.year:
        wanted_years = {str(y) for y in args.year}
        days = sorted(d for d in available_days if d[0:4] in wanted_years)
    else:
        wanted_years = {d[0:4] for d in available_days}
        days = sorted(available_days)

    if not days:
        years_desc = ", ".join(sorted(wanted_years)) if args.year else "any year"
        raise RuntimeError(f"No matching Faraday+Magnetometer day pairs found for {years_desc} in {dscovr_dir}")

    missing_fc = sorted(d for d in mag_files if d[0:4] in wanted_years and d not in fc_files)
    missing_mag = sorted(d for d in fc_files if d[0:4] in wanted_years and d not in mag_files)
    for d in missing_fc:
        print(f"Skipping {d}: magnetometer file present but Faraday Cup file missing")
    for d in missing_mag:
        print(f"Skipping {d}: Faraday Cup file present but magnetometer file missing")

    raw_frames = []
    failed_days = []
    for day in days:
        day_start = pd.Timestamp(f"{day[0:4]}-{day[4:6]}-{day[6:8]}T00:00:00Z")
        day_end = day_start + pd.Timedelta(days=1)
        try:
            merged = load_inputs(fc_files[day], mag_files[day], date_start=day_start, date_end=day_end)
        except Exception as exc:  # noqa: BLE001 -- one bad day shouldn't kill the whole run
            print(f"Skipping {day}: failed to load ({exc})")
            failed_days.append(day)
            continue
        if merged.empty:
            print(f"Skipping {day}: no observations in file")
            failed_days.append(day)
            continue
        raw_frames.append(merged)

    if not raw_frames:
        raise RuntimeError("No days were successfully loaded.")

    full_raw = pd.concat(raw_frames).sort_index()
    full_raw = full_raw[~full_raw.index.duplicated(keep="first")]

    # Compute derived quantities once on the continuous series so rolling
    # windows and diffs span day (and year) boundaries correctly.
    data = add_derived_quantities(full_raw)

    years_sorted = sorted(wanted_years)
    if args.out:
        out_path = Path(args.out).expanduser()
    elif len(years_sorted) == 1:
        year = years_sorted[0]
        out_path = Path(f"dscovr_{year}/dscovr_{year}_core_and_derived.csv")
    else:
        span = f"{years_sorted[0]}_{years_sorted[-1]}"
        out_path = Path(f"dscovr_{span}/dscovr_{span}_core_and_derived.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index_label="time")

    print(f"Years requested: {', '.join(years_sorted)}")
    print(f"Days requested: {len(days)}")
    print(f"Days skipped: {len(failed_days)}{(' -> ' + ', '.join(failed_days)) if failed_days else ''}")
    print(f"Rows written: {len(data)}")
    print(f"Columns: {list(data.columns)}")
    print(f"Output written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
