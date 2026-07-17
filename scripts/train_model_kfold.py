"""Evaluate the linear model with 5-fold cross-validation.

Run from the project root:
    py scripts/train_model_kfold.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: write a file, no window
import matplotlib.pyplot as plt

# Make src/ importable so this script can call the shared model logic
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import cross_validate, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of folds (default: 5).")
    parser.add_argument("--output-dir", default="reports/kfold",
                        help="Where to save metrics and the fold plot.")
    args = parser.parse_args()

    df = load_dataset("features/dataset.csv")
    # Run k-fold cross-validation; fixed seed for reproducibility
    cv = cross_validate(df, n_splits=args.n_splits, random_state=42)

    # Build the metrics report line by line
    lines = [
        f"{args.n_splits}-fold cross-validation  (n = {len(df)} signals)",
        "-" * 48,  # separator rule
        "Per-fold results:",
    ]
    # One line per fold: its RMSE and R2
    for i, (r, r2) in enumerate(zip(cv["fold_rmse"], cv["fold_r2"]), start=1):
        lines.append(f"  fold {i}:  RMSE = {r:.4f}   R2 = {r2:.4f}")
    # Then the aggregate: mean +/- standard deviation across folds
    lines += [
        "",
        f"RMSE:  {cv['rmse_mean']:.4f}  ±  {cv['rmse_std']:.4f}",
        f"R^2 :  {cv['r2_mean']:.4f}  ±  {cv['r2_std']:.4f}",
    ]
    report = "\n".join(lines)
    print(report)

    # Save the report to metrics.txt, creating the folder if needed
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.txt").write_text(report + "\n", encoding="utf-8")

    # bar chart of per-fold RMSE with the mean line
    folds = list(range(1, args.n_splits + 1))  # x-axis: fold numbers
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(folds, cv["fold_rmse"], color="tab:blue", alpha=0.85)  # one bar per fold
    # Dashed red line marks the mean RMSE across folds
    ax.axhline(cv["rmse_mean"], color="crimson", linestyle="--",
               label=f"mean = {cv['rmse_mean']:.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE (test)")
    ax.set_title(f"{args.n_splits}-fold cross-validation — RMSE per fold")
    ax.set_xticks(folds)  # show every fold number on the axis
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="0.92", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "kfold_rmse.png", dpi=150)

    # Report where the outputs were written
    print(f"\nSaved:\n  {out_dir / 'metrics.txt'}\n  {out_dir / 'kfold_rmse.png'}")


# Only run when executed directly, not when imported
if __name__ == "__main__":
    main()
