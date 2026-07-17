"""
Sinusoid recovery tests for calculate_mean_frequency.

Unlike test_mean_frequency.py (which feeds hand-made power matrices directly),
these tests start from an actual WAVEFORM, build a real spectrogram from it via
an FFT, and check the feature extraction recovers the right frequency. This is
much closer to what the real .mat files contain.

Four tests:
  A) Frequency recovery  - 10 pure sines at known frequencies must be recovered.
  B) Amplitude invariance - the centroid is normalised, so amplitude must not
                            change the recovered frequency.
  C) Two-tone weighting   - two sines of known relative power must give the
                            power-weighted average frequency. This is what proves
                            the function computes a CENTROID and not just a peak.
  D) dB path              - converting the spectrogram to dB and back must recover
                            the same frequency (validates P = 10^(dB/10)).

Run from the project root:
    py tests/test_sinusoid_recovery.py
"""

import sys
from pathlib import Path

import numpy as np

# Make src/ importable regardless of where we run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mat_spectrogram_features import calculate_mean_frequency

# Sampling setup for the synthetic signals
FS = 8000.0        # sampling rate (Hz)
DURATION = 1.0     # seconds
NPERSEG = 1024     # FFT window length
NOVERLAP = 512     # window overlap
TOL_HZ = 1.0       # tolerance in Hz (spectral leakage means it is not bit-exact)


def make_spectrogram(freqs_hz, amps):
    """Build a waveform from sinusoids, then its power spectrogram via STFT.

    Returns (freq_vector, power_matrix) with power_matrix shaped frequency x time,
    which is exactly the layout calculate_mean_frequency expects.
    """
    n = int(FS * DURATION)     # total number of samples
    t = np.arange(n) / FS      # time axis in seconds

    # Sum the requested sine waves into one waveform
    x = np.zeros(n)
    for f, a in zip(freqs_hz, amps):
        x += a * np.sin(2 * np.pi * f * t)

    step = NPERSEG - NOVERLAP    # hop between successive windows
    window = np.hanning(NPERSEG)  # taper each window to reduce spectral leakage

    # Slide a window along the signal; one FFT per window = one time frame
    frames = []
    for start in range(0, n - NPERSEG + 1, step):
        segment = x[start:start + NPERSEG] * window
        spectrum = np.fft.rfft(segment)           # real FFT of this window
        frames.append(np.abs(spectrum) ** 2)      # power = |X|^2

    power = np.array(frames).T                     # frequency x time
    freq = np.fft.rfftfreq(NPERSEG, d=1.0 / FS)    # the frequency of each bin (Hz)
    return freq, power


def check(name, got, expected, tol=TOL_HZ):
    ok = abs(got - expected) <= tol   # pass if within tolerance
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: got {got:.3f} Hz, expected {expected:.3f} Hz")
    if not ok:
        raise AssertionError(f"{name} failed (tolerance {tol} Hz)")  # stop on failure


def test_a_frequency_recovery():
    """Ten pure sines at known frequencies must each be recovered."""
    print("A) Frequency recovery - 10 pure sinusoids")
    # For a single tone the centroid should land on that tone's frequency
    for f0 in [200, 400, 600, 800, 1000, 1200, 1500, 1800, 2200, 2600]:
        freq, power = make_spectrogram([f0], [1.0])
        got = calculate_mean_frequency(freq, power, values_are_db=False)
        check(f"  pure sine at {f0} Hz", got, f0)
    print()


def test_b_amplitude_invariance():
    """The centroid is normalised, so amplitude must NOT shift the frequency."""
    print("B) Amplitude invariance - same frequency, different amplitudes")
    f0 = 1000.0
    # Same tone, louder or quieter: recovered frequency must stay 1000 Hz
    for amp in [0.1, 1.0, 5.0, 50.0]:
        freq, power = make_spectrogram([f0], [amp])
        got = calculate_mean_frequency(freq, power, values_are_db=False)
        check(f"  amplitude {amp}", got, f0)
    print()


def test_c_two_tone_weighting():
    """Two tones -> the centroid must be their POWER-weighted average.

    Power scales as amplitude^2, so amplitudes (1, 3) give weights (1, 9).
    This is the test that proves we compute a centroid, not just a peak.
    """
    print("C) Two-tone weighting - proves it is a centroid, not a peak-finder")
    cases = [
        ([300.0, 700.0], [1.0, 1.0], "equal power -> midpoint"),
        ([300.0, 700.0], [1.0, 3.0], "700 Hz stronger -> pulled up"),
        ([300.0, 700.0], [3.0, 1.0], "300 Hz stronger -> pulled down"),
    ]
    for freqs, amps, note in cases:
        freq, power = make_spectrogram(freqs, amps)
        got = calculate_mean_frequency(freq, power, values_are_db=False)

        # Compute the expected centroid by hand, weighting by power (amplitude^2)
        weights = np.array(amps) ** 2                     # power ~ amplitude^2
        expected = float(np.sum(np.array(freqs) * weights) / np.sum(weights))
        check(f"  {freqs} amps {amps}  [{note}]", got, expected)
    print()


def test_d_db_path():
    """Converting to dB and back must recover the same frequency."""
    print("D) dB path - validates P = 10^(dB/10)")
    for f0 in [500.0, 1500.0, 2500.0]:
        freq, power = make_spectrogram([f0], [1.0])

        # Recover the frequency from linear power and from the dB version
        from_linear = calculate_mean_frequency(freq, power, values_are_db=False)
        power_db = 10.0 * np.log10(power + 1e-20)         # guard against log(0)
        from_db = calculate_mean_frequency(freq, power_db, values_are_db=True)

        # Both paths must return the true frequency
        check(f"  {f0} Hz via linear", from_linear, f0)
        check(f"  {f0} Hz via dB    ", from_db, f0)
    print()


def main():
    # Run all four test groups in order
    test_a_frequency_recovery()
    test_b_amplitude_invariance()
    test_c_two_tone_weighting()
    test_d_db_path()
    print("All sinusoid recovery tests passed.")


if __name__ == "__main__":
    main()
