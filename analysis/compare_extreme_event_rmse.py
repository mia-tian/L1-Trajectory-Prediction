"""Compare normal-event vs. extreme-event RMSE across every population-proxy
backend (gbm, xgb, lstm, tcn, dlm), on the same horizon/split/data.

Reuses each backend's own tuned defaults via pipeline/predict_and_backout.py's
run_backend() (same mechanism the pipeline itself uses), so this never
duplicates a second, possibly-drifted copy of any backend's hyperparameters.

"Extreme" follows the convention established earlier this project: the top
quantile (default 90th percentile, i.e. top 10%) by |actual change| --
true_delta = actual - persistence, computed on the TEST set's own real values
only (never touches any model's predictions), symmetric in sign. "Normal" is
everything else (the bottom 90% by that same measure).

Usage
-----
python analysis/compare_extreme_event_rmse.py --horizon 8
python analysis/compare_extreme_event_rmse.py --models gbm xgb dlm --horizon 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("common", "population_proxy", "individual_decay", "data_prep", "analysis", "pipeline"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from predict_and_backout import run_backend  # noqa: E402

DEFAULT_INDIR = "deb_tles_2024"
DEFAULT_SATCAT = "satcat.csv"
ALL_MODELS = ["gbm", "xgb", "lstm", "tcn", "dlm"]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def normal_extreme_rmse(actual: np.ndarray, predicted: np.ndarray, persistence: np.ndarray,
                         extreme_quantile: float) -> dict:
    true_delta = actual - persistence
    threshold = float(np.quantile(np.abs(true_delta), extreme_quantile))
    extreme_mask = np.abs(true_delta) >= threshold
    normal_mask = ~extreme_mask
    return {
        "threshold": threshold,
        "n_normal": int(normal_mask.sum()),
        "n_extreme": int(extreme_mask.sum()),
        "model_normal_rmse": rmse(predicted[normal_mask], actual[normal_mask]),
        "model_extreme_rmse": rmse(predicted[extreme_mask], actual[extreme_mask]),
        "persistence_normal_rmse": rmse(persistence[normal_mask], actual[normal_mask]),
        "persistence_extreme_rmse": rmse(persistence[extreme_mask], actual[extreme_mask]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--indir", default=DEFAULT_INDIR)
    parser.add_argument("--satcat", default=DEFAULT_SATCAT)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--val-date", default="2024-09-01")
    parser.add_argument("--split-date", default="2024-10-15")
    parser.add_argument("--extreme-quantile", type=float, default=0.9,
                         help="Quantile threshold on |actual change| defining 'extreme' (default 0.9 -> top 10%%)")
    parser.add_argument("--output", default="plots/extreme_event_rmse_by_model.png")
    args = parser.parse_args()

    rows = {}
    for model in args.models:
        print(f"\n=== Running backend: {model} ===")
        result = run_backend(model, args.horizon, args.indir, args.satcat, args.year,
                              args.val_date, args.split_date)
        stats = normal_extreme_rmse(result["actual"], result["predicted"], result["persistence"],
                                     args.extreme_quantile)
        rows[model] = stats
        print(f"  n_normal={stats['n_normal']}  n_extreme={stats['n_extreme']}  "
              f"threshold={stats['threshold']:.3f}")
        print(f"  Normal RMSE:  model={stats['model_normal_rmse']:.4f}  "
              f"persistence={stats['persistence_normal_rmse']:.4f}  "
              f"({(1 - stats['model_normal_rmse'] / stats['persistence_normal_rmse']) * 100:+.1f}%)")
        print(f"  Extreme RMSE: model={stats['model_extreme_rmse']:.4f}  "
              f"persistence={stats['persistence_extreme_rmse']:.4f}  "
              f"({(1 - stats['model_extreme_rmse'] / stats['persistence_extreme_rmse']) * 100:+.1f}%)")

    print(f"\n{'model':6s} {'normal RMSE':>12s} {'extreme RMSE':>13s} {'persist normal':>15s} {'persist extreme':>16s}")
    for model, stats in rows.items():
        print(f"{model:6s} {stats['model_normal_rmse']:12.4f} {stats['model_extreme_rmse']:13.4f} "
              f"{stats['persistence_normal_rmse']:15.4f} {stats['persistence_extreme_rmse']:16.4f}")

    # Plot: two panels (normal / extreme), grouped bars per model (model vs. persistence).
    models = list(rows.keys())
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

    ax = axes[0]
    persist_normal = [rows[m]["persistence_normal_rmse"] for m in models]
    model_normal = [rows[m]["model_normal_rmse"] for m in models]
    ax.bar(x - width / 2, persist_normal, width, color="tab:gray", alpha=0.8, label="persistence")
    ax.bar(x + width / 2, model_normal, width, color="tab:blue", alpha=0.9, label="model")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("RMSE")
    ax.set_title(f"Normal events (bottom {args.extreme_quantile * 100:.0f}%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    for xi, (p, m) in enumerate(zip(persist_normal, model_normal)):
        pct = (1 - m / p) * 100
        ax.text(xi, max(p, m) * 1.02, f"{pct:+.1f}%", ha="center", fontsize=8,
                 color="tab:green" if pct > 0 else "tab:red")

    ax = axes[1]
    persist_extreme = [rows[m]["persistence_extreme_rmse"] for m in models]
    model_extreme = [rows[m]["model_extreme_rmse"] for m in models]
    ax.bar(x - width / 2, persist_extreme, width, color="tab:gray", alpha=0.8, label="persistence")
    ax.bar(x + width / 2, model_extreme, width, color="tab:red", alpha=0.9, label="model")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("RMSE")
    ax.set_title(f"Extreme events (top {(1 - args.extreme_quantile) * 100:.0f}%, by |actual change|)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    for xi, (p, m) in enumerate(zip(persist_extreme, model_extreme)):
        pct = (1 - m / p) * 100
        ax.text(xi, max(p, m) * 1.02, f"{pct:+.1f}%", ha="center", fontsize=8,
                 color="tab:green" if pct > 0 else "tab:red")

    fig.suptitle(f"RMSE by event magnitude, {args.horizon}h-ahead, test period ({args.split_date} onward)")
    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    main()
