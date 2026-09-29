"""
Extract the mean-frequency feature from every .mat spectrogram.

Walks a folder of .mat files, computes the power-weighted mean frequency
of each spectrogram, and writes one row per file (the value, how it was
obtained, and a status) to a CSV. A file that fails is recorded as an
error row rather than stopping the run.

The variable keys can be auto-detected, but the project runs pin them
explicitly so the extraction never has to guess.

Run from the project root:
    py scripts/extract_mean_frequencies.py data/raw --freq-key freq
        --spectrogram-key psdx_dB --values-are-db true
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable so this script can call the shared extraction logic
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mat_spectrogram_features import extract_mean_frequencies_from_folder


def main() -> None:
    # Every option below is exposed as a flag, so behaviour changes without editing code
    parser = argparse.ArgumentParser(
        description="Extract the mean-frequency feature from .mat spectrograms."
    )

    parser.add_argument(
        "input_dir",
        nargs="?",              # optional positional: falls back to the default
        default="data/raw",
        help="Folder containing .mat files (default: data/raw).",
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
        action="store_true",    # flag: search sub-folders when present
        help="Search subfolders too.",
    )

    parser.add_argument(
        "--freq-key",
        default=None,           # None: auto-detect the frequency variable
        help="Manual frequency key, for example freq.",
    )

    parser.add_argument(
        "--spectrogram-key",
        default=None,           # None: auto-detect the spectrogram variable
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
        choices=["weighted", "peak"],  # restrict to the two supported methods
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

    # Convert the dB text choice into "auto" / True / False for the function
    if args.values_are_db == "auto":
        values_are_db = "auto"
    elif args.values_are_db == "true":
        values_are_db = True
    else:
        values_are_db = False

    # Run the extraction over every matching file; writes the CSV internally
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

    # Print a per-file summary plus success/error counts
    print(df[["filename", "mean_frequency_hz", "status", "error"]])
    print()
    print(f"Saved CSV to: {args.output}")
    print(f"Successful files: {(df['status'] == 'ok').sum()}")
    print(f"Files with errors: {(df['status'] == 'error').sum()}")


# Only run when executed directly, not when imported
if __name__ == "__main__":
    main()