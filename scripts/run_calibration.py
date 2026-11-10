"""Calibration study: temperature scaling, ECE, reliability diagrams, confidence histograms.

Reads the validation prediction files written by run_baselines.py (or any later model that
saves the same .npz format). Temperature is cross-fitted by prompt group inside validation,
so no row is scored by a calibrator that saw it.

"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration import crossfit_temperature, fit_temperature  # noqa: E402
from src.experiment import load_predictions, load_split, write_json  # noqa: E402
from src.metrics import reliability_figure, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--pattern", default="*__validation.npz")
    ap.add_argument("--skip", nargs="*", default=["class_prior"])
    args = ap.parse_args()
    results = Path(args.results)

    df, _, split = load_split(args.data, args.splits)
    group_of = dict(zip(df.id.astype(str), df.prompt_group))
    report, grouped = {}, defaultdict(list)

    for path in sorted((results / "predictions").glob(args.pattern)):
        model, variant, seed, _ = path.stem.split("__")
        if model in args.skip:
            continue
        ids, p, y = load_predictions(path)
        groups = np.array([group_of[i] for i in ids])
        p_cal, temps = crossfit_temperature(y, p, groups)
        before, after = summarize(y, p), summarize(y, p_cal)
        T_full = fit_temperature(y, p)  # to reuse on the local test set later; fit on validation only
        report[path.stem] = {
            "before": before, "after_crossfit": after,
            "fold_temperatures": temps, "temperature_full_validation": T_full,
            "log_loss_gain": before["log_loss"] - after["log_loss"],
            "ece_change": after["ece_top_label"] - before["ece_top_label"],
        }
        grouped[f"{model}/{variant}"].append(
            (before["log_loss"], after["log_loss"], before["ece_top_label"],
             after["ece_top_label"], T_full))
        if seed.endswith("42"):
            reliability_figure(
                y, {"before": p, "after temp. scaling": p_cal},
                results / "figures" / f"calibration_{model}_{variant}.png",
                title=f"{model}/{variant}")

    summary = {}
    for name, rows in grouped.items():
        a = np.array(rows)
        summary[name] = dict(zip(
            ["log_loss_before", "log_loss_after", "ece_before", "ece_after", "temperature"],
            [float(v) for v in a.mean(axis=0)]))
        print(f"{name:<22} LL {a[:,0].mean():.5f} -> {a[:,1].mean():.5f}   "
              f"ECE {a[:,2].mean():.4f} -> {a[:,3].mean():.4f}   T={a[:,4].mean():.3f}")
    write_json(results / "metrics" / "calibration_study.json",
               {"per_file": report, "summary_mean_over_seeds": summary})
    print("Saved calibration_study.json and reliability figures. Local test NOT scored.")


if __name__ == "__main__":
    main()
