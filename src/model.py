from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURE_COL = "mean_frequency_hz"  # model input X
TARGET_COL = "annoyance_mean"      # model target y


def load_dataset(dataset_csv: str | Path) -> pd.DataFrame:
    """Load the joined modeling dataset and drop rows missing X or y."""
    df = pd.read_csv(dataset_csv)
    # Drop rows without a feature or a label; reset the index for a clean frame
    return df.dropna(subset=[FEATURE_COL, TARGET_COL]).reset_index(drop=True)


def fit_linear_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Split 80/20, fit a single-feature linear regression, and keep predictions."""
    X = df[[FEATURE_COL]].to_numpy()  # 2-D: sklearn expects columns of features
    y = df[TARGET_COL].to_numpy()     # 1-D target

    # Random split; fixed seed makes it reproducible and comparable across runs
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Fit by ordinary least squares on the training set only
    model = LinearRegression().fit(X_train, y_train)

    # Return the model plus data and predictions, so evaluate/plot can reuse them
    return {
        "model": model,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred_train": model.predict(X_train),
        "y_pred_test": model.predict(X_test),
    }


def evaluate(res: dict) -> dict:
    """Compute RMSE (train/test), R2, and a mean-prediction baseline to beat."""
    y_train, y_test = res["y_train"], res["y_test"]

    # RMSE on held-out test data (the honest metric) and on training data
    rmse_test = float(np.sqrt(mean_squared_error(y_test, res["y_pred_test"])))
    rmse_train = float(np.sqrt(mean_squared_error(y_train, res["y_pred_train"])))
    # R2 needs at least 2 points to be defined
    r2_test = float(r2_score(y_test, res["y_pred_test"])) if len(y_test) > 1 else float("nan")

    # Baseline: always predict the training-set mean annoyance.
    baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)  # constant prediction
    rmse_baseline = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))

    model = res["model"]
    return {
        "slope": float(model.coef_[0]),        # a in y = a*x + b
        "intercept": float(model.intercept_),  # b in y = a*x + b
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "rmse_baseline": rmse_baseline,
        "r2_test": r2_test,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def make_plots(df: pd.DataFrame, res: dict, out_dir: str | Path) -> list[Path]:
    """Save (1) scatter of all data + fitted line, (2) predicted vs actual on test."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend: just write files, no window
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # create output folder if needed
    model = res["model"]
    saved = []  # paths of the figures written

    # 1) Data + fitted line
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[FEATURE_COL], df[TARGET_COL], alpha=0.7, label="all signals")
    # Evaluate the fitted line across the frequency range to draw it smoothly
    xs = np.linspace(df[FEATURE_COL].min(), df[FEATURE_COL].max(), 100).reshape(-1, 1)
    ax.plot(xs, model.predict(xs), color="crimson", label="fitted line")
    ax.set_xlabel("Mean frequency (Hz)")
    ax.set_ylabel("Annoyance (mean rating)")
    ax.set_title("Annoyance vs. mean frequency")
    ax.legend()
    fig.tight_layout()
    p1 = out_dir / "fit_scatter.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)  # free memory
    saved.append(p1)

    # 2) Predicted vs actual (test set)
    y_test, y_pred = res["y_test"], res["y_pred_test"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.8)
    # Diagonal y = x is the "perfect prediction" reference, not a fit
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", label="perfect prediction")
    ax.set_xlabel("Actual annoyance")
    ax.set_ylabel("Predicted annoyance")
    ax.set_title("Predicted vs. actual (test set)")
    ax.legend()
    fig.tight_layout()
    p2 = out_dir / "pred_vs_actual.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved.append(p2)

    return saved


def cross_validate(
    df,
    n_splits: int = 5,
    holdout_size: float = 0.20,
    random_state: int = 42,
):
    """Hold out a final test set, run k-fold on the rest, keep the best fold.

    Procedure:
      1. Set aside holdout_size of the signals as a final test set; it is never
         used for training or for choosing a model.
      2. Split the remaining signals into n_splits folds.
      3. Train n_splits times: 4 folds train, 1 fold validates each time.
      4. Select the fold with the LOWEST validation RMSE.
      5. Evaluate that selected model on the untouched holdout set.
      6. The holdout scores are the reported results.
    """
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import KFold, train_test_split

    X = df[[FEATURE_COL]].to_numpy()
    y = df[TARGET_COL].to_numpy()

    # Step 1: separate the final test set from the pool used for k-fold
    X_pool, X_holdout, y_pool, y_holdout = train_test_split(
        X, y, test_size=holdout_size, random_state=random_state
    )

    # Step 2: shuffle the pool then divide it into n_splits folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_models = []   # the model trained in each fold
    fold_rmse = []     # validation RMSE per fold
    fold_r2 = []       # validation R2 per fold
    fold_n_val = []    # size of each validation fold

    # Step 3: each pass trains on the other folds and validates on this one
    for train_idx, val_idx in kf.split(X_pool):
        model = LinearRegression().fit(X_pool[train_idx], y_pool[train_idx])
        y_hat = model.predict(X_pool[val_idx])

        fold_models.append(model)
        fold_rmse.append(float(np.sqrt(mean_squared_error(y_pool[val_idx], y_hat))))
        fold_r2.append(float(r2_score(y_pool[val_idx], y_hat)))
        fold_n_val.append(int(len(val_idx)))

    # Step 4: pick the fold with the lowest validation RMSE
    best_index = int(np.argmin(fold_rmse))
    best_model = fold_models[best_index]

    # Step 5: score the selected model on the holdout it has never seen
    y_hat_holdout = best_model.predict(X_holdout)
    holdout_rmse = float(np.sqrt(mean_squared_error(y_holdout, y_hat_holdout)))
    holdout_r2 = float(r2_score(y_holdout, y_hat_holdout))

    # Step 6: return per-fold detail plus the reported holdout results
    return {
        "n_splits": n_splits,
        "X_holdout": X_holdout,          # holdout features, for plotting
        "y_holdout": y_holdout,          # holdout true annoyance
        "y_pred_holdout": y_hat_holdout, # holdout predictions
        "n_pool": int(len(y_pool)),
        "n_holdout": int(len(y_holdout)),
        "fold_rmse": fold_rmse,
        "fold_r2": fold_r2,
        "fold_n_val": fold_n_val,
        "best_fold": best_index + 1,          # 1-based for reporting
        "best_fold_rmse": fold_rmse[best_index],
        "best_fold_r2": fold_r2[best_index],
        "slope": float(best_model.coef_[0]),      # a in y = a*x + b
        "intercept": float(best_model.intercept_),  # b in y = a*x + b
        "holdout_rmse": holdout_rmse,             # reported result
        "holdout_r2": holdout_r2,                 # reported result
    }