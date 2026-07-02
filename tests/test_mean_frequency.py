"""
Sanity check for calculate_mean_frequency.

Builds synthetic spectrograms with KNOWN centroids and confirms the
function returns the hand-computed answer. Run from the project root:

    python tests/test_mean_frequency.py
"""

import sys
from pathlib import Path

import numpy as np

# Make src/ importable regardless of where we run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mat_spectrogram_features import calculate_mean_frequency


def check(name, got, expected, tol=1e-9):
    ok = np.isclose(got, expected, atol=tol)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: got {got:.4f} Hz, expected {expected:.4f} Hz")
    if not ok:
        raise AssertionError(f"{name} failed")


def main():
    freq = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

    # 1) All power in the 300 Hz bin -> centroid must be exactly 300
    power = np.zeros((5, 3))
    power[2, :] = 1.0
    check("single-bin centroid",
          calculate_mean_frequency(freq, power, values_are_db=False), 300.0)

    # 2) Symmetric weights 1,2,1 around 200 Hz -> 200
    freq3 = np.array([100.0, 200.0, 300.0])
    power = np.array([[1.0], [2.0], [1.0]])
    check("symmetric centroid",
          calculate_mean_frequency(freq3, power, values_are_db=False), 200.0)

    # 3) Asymmetric: (100*1 + 200*1 + 400*2) / 4 = 275
    freqA = np.array([100.0, 200.0, 400.0])
    power = np.array([[1.0], [1.0], [2.0]])
    check("asymmetric weighted mean",
          calculate_mean_frequency(freqA, power, values_are_db=False), 275.0)

    # 4) dB path must match linear path (this tests the dB->power conversion)
    rng = np.random.default_rng(0)
    lin = rng.uniform(0.1, 10.0, size=(5, 4))
    db = 10.0 * np.log10(lin)
    from_linear = calculate_mean_frequency(freq, lin, values_are_db=False)
    from_db = calculate_mean_frequency(freq, db, values_are_db=True)
    check("dB conversion matches linear", from_db, from_linear)

    # 5) fmin/fmax window: keep 200-400, symmetric -> 300
    power = np.ones((5, 3))
    check("fmin/fmax window",
          calculate_mean_frequency(freq, power, values_are_db=False,
                                   fmin=200, fmax=400), 300.0)

    # 6) peak method: strongest bin is 400 Hz every frame -> 400
    power = np.zeros((5, 3))
    power[3, :] = 5.0
    power[0, :] = 1.0
    check("peak method",
          calculate_mean_frequency(freq, power, values_are_db=False,
                                   method="peak"), 400.0)

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()