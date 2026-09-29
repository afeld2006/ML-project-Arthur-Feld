"""
Show what is inside one .mat file.

Lists every variable with its name, shape, type and size, largest numeric
arrays first. This is the exploration tool used at the start of the
project to decide which variable keys to pin (freq and psdx_dB).

Run from the project root:
    py scripts/inspect_mat_file.py data/raw/SignalNum107_A350-941_Approach.mat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable so this script can call the shared inspection function
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mat_spectrogram_features import inspect_mat_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show the variables inside one .mat file."
    )

    # One required argument: the .mat file to look inside
    parser.add_argument(
        "mat_file",
        help="Path to one .mat file.",
    )

    args = parser.parse_args()

    # List the variables in the file (name, shape, dtype, size)
    df = inspect_mat_file(args.mat_file)

    # Print the table without the pandas index for a clean view
    print(df.to_string(index=False))


# Only run when executed directly, not when imported
if __name__ == "__main__":
    main()