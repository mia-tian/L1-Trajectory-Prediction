import argparse
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
for _sub in ("common", "population_proxy", "individual_decay", "data_prep", "analysis"):
    _p = str(_Path(__file__).resolve().parent.parent / _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from traj_predict_tle import get_data_from_tles


def _load_altitude_timeseries(pkl_path):
    with open(pkl_path, "rb") as f:
        t_full, alt_by_obj, satcat = pkl.load(f)
    return t_full, alt_by_obj, satcat


def _align_altitude(t_full, alt_by_obj, t_target):
    index_map = {t: i for i, t in enumerate(t_full)}
    indices = [index_map[t] for t in t_target if t in index_map]
    if len(indices) != len(t_target):
        raise ValueError("Failed to align altitude series to target times.")
    return alt_by_obj[indices, :]


def main():
    parser = argparse.ArgumentParser(
        description="Plot altitude and decay-rate components for one satellite."
    )
    parser.add_argument("--sat-id", type=int, default=None)
    parser.add_argument("--pkl-path", type=str, default="data/example_objs_interp.pkl")
    args = parser.parse_args()

    (
        _,
        _,
        _,
        y_test,
        satcat,
        _,
        _,
        _,
        t_test,
        dalt_mean,
        dalt_std,
        norm_factor_test,
        *_,
    ) = get_data_from_tles()

    if args.sat_id is None:
        sat_id = satcat[0]
    else:
        sat_id = args.sat_id

    if sat_id not in satcat:
        raise ValueError(f"sat_id {sat_id} not found in satcat list.")

    sat_idx = satcat.index(sat_id)

    t_full, alt_by_obj, _ = _load_altitude_timeseries(args.pkl_path)
    alt_aligned = _align_altitude(t_full, alt_by_obj, t_test)
    altitude = alt_aligned[:, sat_idx]

    # Components
    decay_norm = y_test[:, sat_idx]
    scaled_decay = decay_norm * dalt_std[sat_idx] + dalt_mean[sat_idx]
    raw_decay = scaled_decay * norm_factor_test[:, sat_idx]

    # Plot
    plt.figure(figsize=(8, 9))

    plt.subplot(4, 1, 1)
    plt.plot(t_test, altitude, color="tab:blue")
    plt.ylabel("Altitude [km]")
    plt.title(f"Satellite {sat_id}")

    plt.subplot(4, 1, 2)
    plt.plot(t_test, raw_decay, color="tab:orange")
    plt.ylabel("Raw decay rate")

    plt.subplot(4, 1, 3)
    plt.plot(t_test, scaled_decay, color="tab:green")
    plt.ylabel("Scaled decay")

    plt.subplot(4, 1, 4)
    plt.plot(t_test, decay_norm, color="tab:red")
    plt.ylabel("Normalized scaled decay")
    plt.xlabel("Time")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
