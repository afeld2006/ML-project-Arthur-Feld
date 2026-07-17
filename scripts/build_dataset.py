from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable so this script can call the shared logic there
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import build_modeling_dataset

# Default worksheet: the cleaned sheet with 5 participants excluded
CLEANED_SHEET = "Without P. 27, 12, 20, 46, 51"


def main() -> None:
    # Command-line options let the join be reconfigured without editing the code
    parser = argparse.ArgumentParser(
        description="Join mean-frequency features to annoyance labels."
    )
    parser.add_argument(
        "--features",
        default="features/mean_frequency_features.csv",
        help="Feature CSV produced by extract_mean_frequencies.py.",
    )
    parser.add_argument(
        "--labels",
        default="data/labels/PsychoResults_All.xlsx",
        help="Path to the psychoacoustic results workbook.",
    )
    parser.add_argument(
        "--sheet",
        default=CLEANED_SHEET,
        help="Worksheet to use (default: the cleaned 55-participant sheet).",
    )
    parser.add_argument(
        "--output",
        default="features/dataset.csv",
        help="Where to save the joined modeling dataset.",
    )
    parser.add_argument(
        "--exclude-zero",
        action="store_true",  # flag: present means True, absent means False
        help="Treat a rating of 0 as missing instead of a real 0-5 value.",
    )
    args = parser.parse_args()

    # Do the actual join; include_zero is the inverse of the --exclude-zero flag
    ok, unmatched = build_modeling_dataset(
        features_csv=args.features,
        xlsx_path=args.labels,
        sheet_name=args.sheet,
        include_zero=not args.exclude_zero,
    )

    # Save the matched dataset, creating the output folder if needed
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok.to_csv(output, index=False)

    # Print a summary so the join result is visible in the terminal
    print(ok.to_string(index=False))
    print()
    print(f"Saved dataset to: {output}")
    print(f"Matched (feature + label): {len(ok)}")
    print(f"Unmatched features (no label found): {len(unmatched)}")
    # List any feature files that found no label, so mismatches are easy to spot
    if len(unmatched):
        print("\nThese feature files had no matching label:")
        print(unmatched[["filename", "AudioNum", "Phase"]].to_string(index=False))

# Only run main() when this file is executed directly, not when imported
if __name__ == "__main__":
    main()
