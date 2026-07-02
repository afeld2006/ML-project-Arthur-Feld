from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mat_spectrogram_features import extract_mean_frequencies_from_folder


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_dir",
        help="Folder containing .mat files. Use . if the .mat files are in the repo root.",
    )

    parser.add_argument(
        "--output",
        default="features/mean_frequency_features.csv",
        help="Where to save the output CSV.",
    )

    parser.add_argument(
        "--pattern",
        default="*.mat",
        help="File pattern to search for.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subfolders too.",
    )

    parser.add_argument(
        "--freq-key",
        default=None,
        help="Manual frequency key, for example freq.",
    )

    parser.add_argument(
        "--spectrogram-key",
        default=None,
        help="Manual spectrogram key, for example psdx_dB.",
    )

    parser.add_argument(
        "--fmin",
        type=float,
        default=None,
        help="Minimum frequency in Hz.",
    )

    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Maximum frequency in Hz.",
    )

    parser.add_argument(
        "--method",
        choices=["weighted", "peak"],
        default="weighted",
        help="Mean frequency method.",
    )

    parser.add_argument(
        "--values-are-db",
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether spectrogram values are in dB.",
    )

    args = parser.parse_args()

    if args.values_are_db == "auto":
        values_are_db = "auto"
    elif args.values_are_db == "true":
        values_are_db = True
    else:
        values_are_db = False

    df = extract_mean_frequencies_from_folder(
        input_dir=args.input_dir,
        pattern=args.pattern,
        recursive=args.recursive,
        output_csv=args.output,
        freq_key=args.freq_key,
        spectrogram_key=args.spectrogram_key,
        values_are_db=values_are_db,
        fmin=args.fmin,
        fmax=args.fmax,
        method=args.method,
    )

    print(df[["filename", "mean_frequency_hz", "status", "error"]])
    print()
    print(f"Saved CSV to: {args.output}")
    print(f"Successful files: {(df['status'] == 'ok').sum()}")
    print(f"Files with errors: {(df['status'] == 'error').sum()}")


if __name__ == "__main__":
    main()
