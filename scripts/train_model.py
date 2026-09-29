"""
Train and evaluate the MVP model on one train/test split.

Fits the single-feature linear regression (annoyance ~ mean frequency),
scores it on the held-out fraction against the predict-the-mean baseline,
and writes metrics.txt plus the two standard figures into the output
folder.

Changing --test-size is how the split experiment was run, one output
folder per split:

Run from the project root:
    py scripts/train_model.py --test-size 0.20 --output-dir reports/split_8020
    py scripts/train_model.py --test-size 0.30 --output-dir reports/split_7030
    py scripts/train_model.py --test-size 0.45 --output-dir reports/split_5545
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable so this script can call the shared model logic
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import evaluate, fit_linear_model, load_dataset, make_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the MVP linear regression: annoyance ~ mean frequency."
    )
    parser.add_argument("--dataset", default="features/dataset.csv",
                        help="Joined dataset from build_dataset.py.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test fraction for the train/test split.")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Seed for a reproducible split.")
    parser.add_argument("--output-dir", default="reports",
                        help="Where to save plots and metrics.")
    args = parser.parse_args()

    # Pipeline: load data, fit the model, score it, and make the plots
    df = load_dataset(args.dataset)
    res = fit_linear_model(df, test_size=args.test_size,
                           random_state=args.random_state)
    m = evaluate(res)
    plots = make_plots(df, res, args.output_dir)

    # Build the metrics report line by line for printing and saving
    lines = [
        "MVP linear model: annoyance ~ mean_frequency_hz",
        "-" * 48,  # separator rule
        f"samples: {len(df)}  (train {m['n_train']}, test {m['n_test']})",
        f"fitted:  annoyance = {m['slope']:.6f} * freq + {m['intercept']:.4f}",
        "",
        f"RMSE (test):      {m['rmse_test']:.4f}",
        f"RMSE (train):     {m['rmse_train']:.4f}",
        f"RMSE (baseline):  {m['rmse_baseline']:.4f}   <- predicting the mean",
        f"R^2  (test):      {m['r2_test']:.4f}",
    ]
    # Verdict: did the model beat the mean-predicting baseline?
    verdict = ("Beats the baseline." if m["rmse_test"] < m["rmse_baseline"]
               else "Does NOT beat the baseline (single feature is weak).")
    lines += ["", verdict]
    report = "\n".join(lines)
    print(report)

    # Save the same report to metrics.txt, creating the folder if needed
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.txt").write_text(report + "\n", encoding="utf-8")
    # List every file written (the metrics file plus the plots)
    print("\nSaved:")
    for p in [out_dir / "metrics.txt", *plots]:
        print(f"  {p}")


# Only run when executed directly, not when imported
if __name__ == "__main__":
    main()