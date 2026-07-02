from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mat_spectrogram_features import inspect_mat_file


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mat_file",
        help="Path to one .mat file.",
    )

    args = parser.parse_args()

    df = inspect_mat_file(args.mat_file)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
