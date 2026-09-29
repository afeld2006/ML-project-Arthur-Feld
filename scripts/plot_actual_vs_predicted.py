"""
Plot actual vs. predicted annoyance against mean frequency, for the TEST SET only.

Black dots = real annoyance scores (the labels y) for the held-out test signals.
Blue dots  = the model's predictions (which lie on the fitted line).
A thin vertical line joins each pair, showing the residual (the error).

Run from the project root, e.g.:
    py scripts/plot_actual_vs_predicted.py --test-size 0.30 --output-dir reports/split_7030
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: write a file, no window
import matplotlib.pyplot as plt

# Make src/ importable so we can reuse the model-fitting logic
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import fit_linear_model, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot real vs. predicted annoyance on the test set."
    )
    parser.add_argument("--test-size", type=float, default=0.20,
                        help="Test fraction for the split (e.g. 0.20, 0.30, 0.45).")
    parser.add_argument("--output-dir", default="reports",
                        help="Folder to save the plot in.")
    args = parser.parse_args()

    df = load_dataset("features/dataset.csv")
    # Refit with the requested split; fixed seed matches the other scripts
    res = fit_linear_model(df, test_size=args.test_size, random_state=42)

    # Use ONLY the held-out test set (the points the model never saw)
    x = res["X_test"].ravel()          # mean frequency of each test signal
    y_true = res["y_test"]             # real annoyance
    y_pred = res["y_pred_test"]        # model prediction

    # Build a "8020"-style tag from the split, used in the title and filename
    pct = round((1 - args.test_size) * 100)
    tag = f"{pct}{100 - pct}"

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Vertical grey line per point: shows the residual (real vs. predicted gap)
    for xi, yt, yp in zip(x, y_true, y_pred):
        ax.plot([xi, xi], [yt, yp], color="0.8", linewidth=0.8, zorder=1)

    # Black dots: real annoyance; blue dots: predictions (drawn on top, zorder=3)
    ax.scatter(x, y_true, color="black", s=40, zorder=3,
               label="Real annoyance (y)")
    ax.scatter(x, y_pred, color="tab:blue", s=40, zorder=3,
               label="Predicted annoyance")

    ax.set_xlabel("Mean frequency (Hz)")
    ax.set_ylabel("Annoyance (0-5)")
    ax.set_title(f"Real vs. predicted annoyance : test set ({tag} split)")
    ax.legend(frameon=False)
    ax.grid(True, color="0.92", linewidth=0.6)
    fig.tight_layout()

    # Save into the chosen folder, creating it if needed; name encodes the split
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"actual_vs_predicted_test_{tag}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


# Only run when executed directly, not when imported
if __name__ == "__main__":
    main()
