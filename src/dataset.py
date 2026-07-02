from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PHASES = ("Approach", "Flyover", "Recede")
_FILENAME_RE = re.compile(r"SignalNum(\d+).*?(Approach|Flyover|Recede)", re.IGNORECASE)


def parse_signal_and_phase(filename: str) -> tuple[int | None, str | None]:
    """Pull (SignalNum, Phase) out of a .mat filename like
    'SignalNum107_A350-941_Approach.mat'. Returns (None, None) if no match."""
    m = _FILENAME_RE.search(str(filename))
    if not m:
        return None, None
    phase = m.group(2).capitalize()
    return int(m.group(1)), phase


def load_and_aggregate_labels(
    xlsx_path: str | Path,
    sheet_name: str,
    include_zero: bool = True,
) -> pd.DataFrame:
    """Load the psychoacoustic sheet, compute the per-row mean annoyance across
    participant columns, then average any repeated (AudioNum, Phase) presentations."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    participant_cols = [c for c in df.columns if isinstance(c, int)]

    ratings = df[participant_cols]
    if not include_zero:
        ratings = ratings.replace(0, pd.NA)

    df = df.assign(annoyance_mean=ratings.mean(axis=1))

    agg = (
        df.groupby(["AudioNum", "Phase"], as_index=False)
        .agg(
            annoyance_mean=("annoyance_mean", "mean"),
            n_presentations=("annoyance_mean", "size"),
        )
    )
    return agg


def build_modeling_dataset(
    features_csv: str | Path,
    xlsx_path: str | Path,
    sheet_name: str,
    include_zero: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join mean-frequency features to aggregated annoyance labels on
    (SignalNum, Phase). Returns (merged_ok, unmatched_features)."""
    feats = pd.read_csv(features_csv)
    feats = feats[feats["status"] == "ok"].copy()

    parsed = feats["filename"].apply(parse_signal_and_phase)
    feats["AudioNum"] = parsed.apply(lambda t: t[0])
    feats["Phase"] = parsed.apply(lambda t: t[1])

    labels = load_and_aggregate_labels(xlsx_path, sheet_name, include_zero=include_zero)

    merged = feats.merge(labels, on=["AudioNum", "Phase"], how="left")

    ok = merged[merged["annoyance_mean"].notna()].copy()
    unmatched = merged[merged["annoyance_mean"].isna()].copy()

    cols = ["filename", "AudioNum", "Phase", "mean_frequency_hz",
            "annoyance_mean", "n_presentations"]
    ok = ok[cols].sort_values(["AudioNum", "Phase"]).reset_index(drop=True)
    return ok, unmatched