"""
Plot actual vs. predicted annoyance against mean frequency.

Black dots = real annoyance scores (the labels y).
Blue dots  = the model's predictions (which lie on the fitted line).
A thin vertical line joins each pair, showing the residual (the error).

Run from the project root:
    py scripts/plot_actual_vs_predicted.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import FEATURE_COL, TARGET_COL, fit_linear_model, load_dataset


def main() -> None:
    df = load_dataset("features/dataset.csv")
    res = fit_linear_model(df, test_size=0.2, random_state=42)
    model = res["model"]

    # Predict for EVERY signal (so the plot shows the full dataset)
    X = df[[FEATURE_COL]].to_numpy()
    x = X.ravel()
    y_true = df[TARGET_COL].to_numpy()
    y_pred = model.predict(X)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # residual lines first (drawn underneath the dots)
    for xi, yt, yp in zip(x, y_true, y_pred):
        ax.plot([xi, xi], [yt, yp], color="0.8", linewidth=0.8, zorder=1)

    # real annoyance scores = black dots
    ax.scatter(x, y_true, color="black", s=30, zorder=3,
               label="Real annoyance (y)")
    # predictions = blue dots (they lie on the fitted line)
    ax.scatter(x, y_pred, color="tab:blue", s=30, zorder=3,
               label="Predicted annoyance")

    ax.set_xlabel("Mean frequency (Hz)")
    ax.set_ylabel("Annoyance (0-5)")
    ax.set_title("Real vs. predicted annoyance across mean frequency")
    ax.legend(frameon=False)
    ax.grid(True, color="0.92", linewidth=0.6)
    fig.tight_layout()

    out = Path("reports/actual_vs_predicted_over_freq.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()