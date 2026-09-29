"""
Plot the synthetic signals used in test_sinusoid_recovery.py.

Produces figures showing, for test groups A to C, the waveform on the left and
its power spectrum on the right, with the recovered mean frequency marked. These
are visual companions to the tests; they do not verify anything themselves.
Group D only re-checks the dB conversion, so it has no figure of its own.

Run from the project root:
    py tests/plot_test_signals.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: write files, no window
import matplotlib.pyplot as plt
import numpy as np

# Make src/ importable so we reuse the same feature function the tests use
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mat_spectrogram_features import calculate_mean_frequency

# Same sampling setup as test_sinusoid_recovery.py, so the plots match the tests
FS = 8000.0        # sampling rate (Hz)
DURATION = 1.0     # seconds
NPERSEG = 1024     # FFT window length
NOVERLAP = 512     # window overlap

OUT_DIR = PROJECT_ROOT / "tests" / "figures"  # where the figures are saved
PLOT_MS = 20.0     # how many milliseconds of waveform to display
FMAX_PLOT = 3000.0  # upper frequency shown on the spectrum plots


def make_signal(freqs_hz, amps):
    """Build the summed sine waveform and its time axis."""
    n = int(FS * DURATION)
    t = np.arange(n) / FS
    x = np.zeros(n)
    for f, a in zip(freqs_hz, amps):
        x += a * np.sin(2 * np.pi * f * t)
    return t, x


def make_spectrogram(x):
    """Turn a waveform into a power spectrogram via STFT; same as in the tests."""
    n = len(x)
    step = NPERSEG - NOVERLAP
    window = np.hanning(NPERSEG)  # taper to reduce spectral leakage

    frames = []
    for start in range(0, n - NPERSEG + 1, step):
        segment = x[start:start + NPERSEG] * window
        spectrum = np.fft.rfft(segment)
        frames.append(np.abs(spectrum) ** 2)   # power = |X|^2

    power = np.array(frames).T                  # frequency x time
    freq = np.fft.rfftfreq(NPERSEG, d=1.0 / FS)
    return freq, power


def plot_case(ax_wave, ax_spec, freqs_hz, amps, title):
    """Draw one case: waveform on the left, spectrum plus centroid on the right."""
    t, x = make_signal(freqs_hz, amps)
    freq, power = make_spectrogram(x)
    centroid = calculate_mean_frequency(freq, power, values_are_db=False)

    # Left: a short slice of the waveform, so individual cycles are visible
    n_show = int(FS * PLOT_MS / 1000.0)
    ax_wave.plot(t[:n_show] * 1000.0, x[:n_show], color="tab:blue", linewidth=1.2)
    ax_wave.set_xlabel("Time (ms)")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title(f"{title} : waveform")
    ax_wave.grid(True, color="0.92", linewidth=0.6)

    # Right: the time-averaged spectrum, with the recovered centroid marked
    spectrum = power.mean(axis=1)               # average over time frames
    keep = freq <= FMAX_PLOT                    # zoom on the interesting range
    ax_spec.plot(freq[keep], spectrum[keep], color="tab:blue", linewidth=1.2)
    ax_spec.axvline(centroid, color="crimson", linestyle="--",
                    label=f"mean frequency = {centroid:.1f} Hz")
    # Mark each true tone so the reader can compare it with the centroid
    for f in freqs_hz:
        ax_spec.axvline(f, color="0.55", linestyle=":", linewidth=1.0)
    ax_spec.set_xlabel("Frequency (Hz)")
    ax_spec.set_ylabel("Power")
    ax_spec.set_title(f"{title} : spectrum")
    ax_spec.legend(frameon=False, fontsize=9)
    ax_spec.grid(True, color="0.92", linewidth=0.6)

    return centroid


def figure_single_tone():
    """Test A: one pure sine; the centroid must land on that frequency."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, f0 in enumerate([400.0, 1200.0]):
        plot_case(axes[row, 0], axes[row, 1], [f0], [1.0], f"Pure sine {f0:.0f} Hz")
    fig.suptitle("Test A : frequency recovery from a single tone", fontsize=13)
    fig.tight_layout()
    path = OUT_DIR / "test_A_single_tone.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_amplitude_invariance():
    """Test B: same tone at two amplitudes; the centroid must not move."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, amp in enumerate([1.0, 5.0]):
        plot_case(axes[row, 0], axes[row, 1], [1000.0], [amp],
                  f"1000 Hz, amplitude {amp:g}")
    fig.suptitle("Test B : amplitude invariance; louder signal, same frequency",
                 fontsize=13)
    fig.tight_layout()
    path = OUT_DIR / "test_B_amplitude_invariance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_two_tone():
    """Test C: two tones; the centroid is their power-weighted average."""
    cases = [
        ([300.0, 700.0], [1.0, 1.0], "Equal power"),
        ([300.0, 700.0], [1.0, 3.0], "700 Hz three times stronger"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, (freqs, amps, label) in enumerate(cases):
        plot_case(axes[row, 0], axes[row, 1], freqs, amps, label)
    fig.suptitle("Test C : two tones; the centroid sits between them, "
                 "pulled toward the stronger", fontsize=13)
    fig.tight_layout()
    path = OUT_DIR / "test_C_two_tone.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)  # create the figures folder
    paths = [
        figure_single_tone(),
        figure_amplitude_invariance(),
        figure_two_tone(),
    ]
    print("Saved:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()