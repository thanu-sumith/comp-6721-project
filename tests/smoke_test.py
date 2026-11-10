"""Pipeline smoke test on SYNTHETIC data. Verifies the code runs end to end;
the numbers it prints mean nothing about the real project. Run: python tests/smoke_test.py"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORDS = ("the model answer should explain because however example code list result data "
         "sorry cannot help sure here steps first second note important simple context").split()


def synth(n=2400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        pr = " ".join(rng.choice(WORDS, rng.integers(4, 20)))
        la, lb = rng.integers(3, 120, 2)
        ra = " ".join(rng.choice(WORDS, la)) + ("\n- item\n- item" if rng.random() < .3 else "")
        rb = " ".join(rng.choice(WORDS, lb)) + ("\n- item\n- item" if rng.random() < .3 else "")
        score = np.log(la / lb) + rng.normal(0, 1.0)
        lab = 2 if abs(score) < 0.35 else (0 if score > 0 else 1)
        rows.append({"id": 1000 + i, "model_a": "x", "model_b": "y",
                     "prompt": json.dumps([pr + f" {i % 600}"]), "response_a": json.dumps([ra]),
                     "response_b": json.dumps([rb]), "winner_model_a": int(lab == 0),
                     "winner_model_b": int(lab == 1), "winner_tie": int(lab == 2)})
    return pd.DataFrame(rows)


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-1800:])
    if r.returncode:
        print(r.stderr[-3000:])
        raise SystemExit(f"FAILED: {args}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from src.baseline import load_data, saved_split
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = tmp / "train.csv"
        synth().to_csv(csv, index=False)
        df, fp = load_data(csv)
        saved_split(df, fp, tmp / "splits")
        common = ["--data", csv, "--splits", tmp / "splits", "--results", tmp / "results",
                  "--cache", tmp / "cache"]
        run("scripts/run_baselines.py", *common, "--seeds", 42, 43)
        run("scripts/run_calibration.py", *common[:6])
        run("scripts/run_data_efficiency.py", *common, "--seeds", 42, 43,
            "--fractions", 0.25, 1.0, "--clean-fractions", 0.0, 0.1)
        print("Outputs:", sorted(p.name for p in (tmp / "results").rglob("*") if p.is_file())[:30])
        print("SMOKE TEST PASSED (synthetic data; ignore the numbers)")
