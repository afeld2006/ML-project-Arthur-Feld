"""Evaluate the linear model with a holdout set plus 5-fold cross-validation.

Procedure:
  1. Hold out 20% of the signals as a final test set.
  2. Split the remaining 80% into 5 folds.
  3. Train 5 times: 4 folds train, 1 fold validates.
  4. Select the fold with the lowest validation RMSE.
  5. Evaluate that model on the untouched 20%.
  6. Report those RMSE and R2 values as the results.

Writes metrics.txt (per-fold results plus the final scores), the per-fold
RMSE bar chart, and the real vs. predicted plot on the holdout signals.

Run from the project root:
    py scripts/train_model_kfold.py
"""

from __future__ import annotations

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
    parser = argparse.ArgumentParser(
        description="Evaluate the MVP model with a holdout set plus k-fold selection."
    )
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of folds on the 80 percent pool (default: 5).")
    parser.add_argument("--holdout-size", type=float, default=0.20,
                        help="Fraction held out as the final test set (default: 0.20).")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Seed for a reproducible holdout and fold split.")
    parser.add_argument("--output-dir", default="reports/kfold",
                        help="Where to save metrics and the plots.")
    args = parser.parse_args()

    df = load_dataset("features/dataset.csv")
    # Run the full procedure: holdout, folds, best-fold selection, final scoring
    cv = cross_validate(
        df,
        n_splits=args.n_splits,
        holdout_size=args.holdout_size,
        random_state=args.random_state,
    )

    # Build the report line by line
    lines = [
        f"{args.n_splits}-fold with holdout  (n = {len(df)} signals)",
        "=" * 52,
        f"pool used for folds: {cv['n_pool']} signals",
        f"final holdout set:   {cv['n_holdout']} signals (never seen)",
        "",
        "Validation results per fold:",
    ]
    # One line per fold; mark the selected one
    for i, (r, r2, n_val) in enumerate(
        zip(cv["fold_rmse"], cv["fold_r2"], cv["fold_n_val"]), start=1
    ):
        marker = "  <= best" if i == cv["best_fold"] else ""
        lines.append(f"  fold {i} (val n={n_val}):  RMSE = {r:.4f}   R2 = {r2:.4f}{marker}")

    # The selected model and its final scores on the untouched holdout
    lines += [
        "",
        f"Selected fold: {cv['best_fold']} (lowest validation RMSE)",
        f"fitted:  annoyance = {cv['slope']:.6f} * freq + {cv['intercept']:.4f}",
        "",
        "FINAL RESULTS on the held-out 20 percent:",
        f"  RMSE:      {cv['holdout_rmse']:.4f}",
        f"  R^2 :      {cv['holdout_r2']:.4f}",
    ]
    report = "\n".join(lines)
    print(report)

    # Save the report, creating the output folder if needed
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.txt").write_text(report + "\n", encoding="utf-8")

    # Plot 1: bar chart of validation RMSE per fold; selected fold highlighted
    folds = list(range(1, args.n_splits + 1))
    colors = ["tab:blue"] * args.n_splits
    colors[cv["best_fold"] - 1] = "tab:green"  # highlight the chosen fold

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(folds, cv["fold_rmse"], color=colors, alpha=0.85)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Validation RMSE")
    ax.set_title(f"Validation RMSE per fold; fold {cv['best_fold']} selected")
    ax.set_xticks(folds)  # show every fold number
    ax.grid(True, axis="y", color="0.92", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "kfold_rmse.png", dpi=150)
    plt.close(fig)

    # Plot 2: real vs predicted annoyance on the holdout signals
    x = cv["X_holdout"].ravel()    # mean frequency of each holdout signal
    y_true = cv["y_holdout"]       # real annoyance
    y_pred = cv["y_pred_holdout"]  # prediction from the selected model

    fig, ax = plt.subplots(figsize=(8, 5.5))
    # Vertical grey line per point: shows the residual (real vs predicted gap)
    for xi, yt, yp in zip(x, y_true, y_pred):
        ax.plot([xi, xi], [yt, yp], color="0.8", linewidth=0.8, zorder=1)

    # Black dots: real annoyance; blue dots: predictions (drawn on top)
    ax.scatter(x, y_true, color="black", s=40, zorder=3,
               label="Real annoyance (y)")
    ax.scatter(x, y_pred, color="tab:blue", s=40, zorder=3,
               label="Predicted annoyance")

    ax.set_xlabel("Mean frequency (Hz)")
    ax.set_ylabel("Annoyance (0-5)")
    ax.set_title(f"Real vs. predicted annoyance : holdout set (fold {cv['best_fold']})")
    ax.legend(frameon=False)
    ax.grid(True, color="0.92", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "actual_vs_predicted_holdout.png", dpi=150)
    plt.close(fig)

    # Report where the outputs were written
    print(f"\nSaved:\n  {out_dir / 'metrics.txt'}"
          f"\n  {out_dir / 'kfold_rmse.png'}"
          f"\n  {out_dir / 'actual_vs_predicted_holdout.png'}")


# Only run when executed directly, not when imported
if __name__ == "__main__":
    main()
