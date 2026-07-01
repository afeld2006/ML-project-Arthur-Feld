from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat


def clean_mat_dict(mat: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def load_any_mat(mat_path: str | Path) -> dict[str, Any]:
    """
    Load a MATLAB .mat file.

    Works for normal .mat files through scipy.
    Tries h5py if the file is MATLAB v7.3.
    """
    mat_path = Path(mat_path)

    try:
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        return clean_mat_dict(mat)

    except NotImplementedError:
        return load_hdf5_mat(mat_path)

    except ValueError as exc:
        message = str(exc).lower()

        if "unknown mat file type" in message or "please use hdf reader" in message:
            return load_hdf5_mat(mat_path)

        raise


def load_hdf5_mat(mat_path: Path) -> dict[str, Any]:
    """
    Fallback loader for MATLAB v7.3 files.
    """
    import h5py

    output = {}

    def visit_group(group, prefix=""):
        for key, item in group.items():
            full_key = f"{prefix}/{key}" if prefix else key

            if isinstance(item, h5py.Dataset):
                arr = np.array(item)

                if np.issubdtype(arr.dtype, np.number):
                    output[full_key] = np.squeeze(arr)

            elif isinstance(item, h5py.Group):
                visit_group(item, full_key)

    with h5py.File(mat_path, "r") as h5_file:
        visit_group(h5_file)

    return output


def is_numeric_array(x: Any) -> bool:
    try:
        arr = np.asarray(x)
    except Exception:
        return False

    return np.issubdtype(arr.dtype, np.number)


def looks_like_frequency_vector(arr: np.ndarray) -> bool:
    arr = np.squeeze(arr)

    if arr.ndim != 1:
        return False

    if arr.size < 2:
        return False

    if not np.all(np.isfinite(arr)):
        return False

    diffs = np.diff(arr)

    return bool(np.all(diffs >= 0) and np.min(arr) >= 0)


def find_frequency_vector(
    mat: dict[str, Any],
    freq_key: str | None = None,
) -> tuple[str, np.ndarray]:
    """
    Find the frequency vector inside a .mat file.
    """

    if freq_key is not None:
        freq = np.squeeze(np.asarray(mat[freq_key], dtype=float))
        return freq_key, freq

    common_names = [
        "freq",
        "freqs",
        "frequency",
        "frequencies",
        "f",
        "F",
        "freq_vector",
        "frequency_vector",
    ]

    for key in common_names:
        if key in mat and is_numeric_array(mat[key]):
            candidate = np.squeeze(np.asarray(mat[key], dtype=float))

            if looks_like_frequency_vector(candidate):
                return key, candidate

    candidates = []

    for key, value in mat.items():
        if not is_numeric_array(value):
            continue

        candidate = np.squeeze(np.asarray(value, dtype=float))

        if looks_like_frequency_vector(candidate):
            candidates.append((key, candidate))

    if not candidates:
        raise ValueError("Could not automatically find a frequency vector.")

    candidates.sort(key=lambda item: item[1].size, reverse=True)

    return candidates[0]


def find_spectrogram_matrix(
    mat: dict[str, Any],
    freq: np.ndarray,
    spectrogram_key: str | None = None,
) -> tuple[str, np.ndarray]:
    """
    Find the spectrogram matrix inside a .mat file.

    Returns a matrix with shape frequency x time.
    """

    freq_len = len(freq)

    if spectrogram_key is not None:
        spectrogram = np.squeeze(np.asarray(mat[spectrogram_key], dtype=float))

        if spectrogram.shape[0] == freq_len:
            return spectrogram_key, spectrogram

        if spectrogram.shape[1] == freq_len:
            return spectrogram_key, spectrogram.T

        raise ValueError(
            f"Spectrogram shape {spectrogram.shape} does not match frequency length {freq_len}."
        )

    possible_name_parts = [
        "psdx",
        "sxx",
        "pxx",
        "spectrogram",
        "spec",
        "stft",
        "psd",
        "power",
        "db",
    ]

    candidates = []

    for key, value in mat.items():
        if not is_numeric_array(value):
            continue

        arr = np.squeeze(np.asarray(value, dtype=float))

        if arr.ndim != 2:
            continue

        if arr.shape[0] == freq_len:
            spectrogram = arr
        elif arr.shape[1] == freq_len:
            spectrogram = arr.T
        else:
            continue

        name_score = 0
        key_lower = key.lower()

        for part in possible_name_parts:
            if part in key_lower:
                name_score += 1

        candidates.append((key, spectrogram, name_score, spectrogram.size))

    if not candidates:
        raise ValueError("Could not automatically find a spectrogram matrix.")

    candidates.sort(key=lambda item: (item[2], item[3]), reverse=True)

    key, spectrogram, _, _ = candidates[0]

    return key, spectrogram


def detect_db_values(spectrogram: np.ndarray, spectrogram_key: str) -> bool:
    """
    Guess whether values are in dB.
    """

    if "db" in spectrogram_key.lower():
        return True

    finite_values = spectrogram[np.isfinite(spectrogram)]

    if finite_values.size == 0:
        raise ValueError("Spectrogram has no finite values.")

    # Linear power should not be negative.
    if np.min(finite_values) < 0:
        return True

    return False


def calculate_mean_frequency(
    freq: np.ndarray,
    spectrogram: np.ndarray,
    values_are_db: bool = True,
    fmin: float | None = None,
    fmax: float | None = None,
    method: str = "weighted",
) -> float:
    """
    Calculate mean frequency from one spectrogram.

    method="weighted":
        Power-weighted mean frequency across the whole spectrogram.

    method="peak":
        Strongest frequency at each time step, then averaged.
    """

    freq = np.asarray(freq, dtype=float)
    spectrogram = np.asarray(spectrogram, dtype=float)

    if spectrogram.shape[0] != len(freq):
        raise ValueError("Spectrogram must have shape frequency x time.")

    mask = np.ones_like(freq, dtype=bool)

    if fmin is not None:
        mask &= freq >= fmin

    if fmax is not None:
        mask &= freq <= fmax

    freq = freq[mask]
    spectrogram = spectrogram[mask, :]

    if freq.size == 0:
        raise ValueError("No frequency bins remain after fmin/fmax filtering.")

    if values_are_db:
        power = 10 ** (spectrogram / 10)
    else:
        power = spectrogram.copy()

    power = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)
    power = np.maximum(power, 0.0)

    if method == "weighted":
        total_power = np.sum(power)

        if total_power <= 0:
            raise ValueError("Total power is zero.")

        mean_frequency = np.sum(freq[:, None] * power) / total_power

        return float(mean_frequency)

    if method == "peak":
        peak_indices = np.argmax(power, axis=0)
        peak_frequencies = freq[peak_indices]

        return float(np.mean(peak_frequencies))

    raise ValueError("method must be 'weighted' or 'peak'.")


def extract_mean_frequency_from_mat(
    mat_path: str | Path,
    freq_key: str | None = None,
    spectrogram_key: str | None = None,
    values_are_db: bool | str = "auto",
    fmin: float | None = None,
    fmax: float | None = None,
    method: str = "weighted",
) -> dict[str, Any]:
    """
    Extract one mean-frequency value from one .mat file.
    """

    mat_path = Path(mat_path)
    mat = load_any_mat(mat_path)

    detected_freq_key, freq = find_frequency_vector(mat, freq_key=freq_key)

    detected_spectrogram_key, spectrogram = find_spectrogram_matrix(
        mat,
        freq=freq,
        spectrogram_key=spectrogram_key,
    )

    if values_are_db == "auto":
        db_used = detect_db_values(spectrogram, detected_spectrogram_key)
    else:
        db_used = bool(values_are_db)

    mean_frequency_hz = calculate_mean_frequency(
        freq=freq,
        spectrogram=spectrogram,
        values_are_db=db_used,
        fmin=fmin,
        fmax=fmax,
        method=method,
    )

    return {
        "filename": mat_path.name,
        "filepath": str(mat_path),
        "mean_frequency_hz": mean_frequency_hz,
        "method": method,
        "fmin_hz": fmin,
        "fmax_hz": fmax,
        "values_are_db": db_used,
        "freq_key": detected_freq_key,
        "spectrogram_key": detected_spectrogram_key,
        "n_frequency_bins": len(freq),
        "n_time_bins": spectrogram.shape[1],
        "status": "ok",
        "error": "",
    }


def extract_mean_frequencies_from_folder(
    input_dir: str | Path,
    pattern: str = "*.mat",
    recursive: bool = False,
    output_csv: str | Path | None = None,
    freq_key: str | None = None,
    spectrogram_key: str | None = None,
    values_are_db: bool | str = "auto",
    fmin: float | None = None,
    fmax: float | None = None,
    method: str = "weighted",
) -> pd.DataFrame:
    """
    Extract mean frequencies from every .mat file in a folder.
    """

    input_dir = Path(input_dir)

    if recursive:
        mat_files = sorted(input_dir.rglob(pattern))
    else:
        mat_files = sorted(input_dir.glob(pattern))

    if not mat_files:
        raise FileNotFoundError(f"No files matching {pattern} found in {input_dir}")

    rows = []

    for mat_file in mat_files:
        try:
            row = extract_mean_frequency_from_mat(
                mat_path=mat_file,
                freq_key=freq_key,
                spectrogram_key=spectrogram_key,
                values_are_db=values_are_db,
                fmin=fmin,
                fmax=fmax,
                method=method,
            )

        except Exception as exc:
            row = {
                "filename": mat_file.name,
                "filepath": str(mat_file),
                "mean_frequency_hz": np.nan,
                "method": method,
                "fmin_hz": fmin,
                "fmax_hz": fmax,
                "values_are_db": values_are_db,
                "freq_key": freq_key,
                "spectrogram_key": spectrogram_key,
                "n_frequency_bins": np.nan,
                "n_time_bins": np.nan,
                "status": "error",
                "error": str(exc),
            }

        rows.append(row)

    df = pd.DataFrame(rows)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df


def inspect_mat_file(mat_path: str | Path) -> pd.DataFrame:
    """
    Show variables inside one .mat file.
    """

    mat = load_any_mat(mat_path)

    rows = []

    for key, value in mat.items():
        arr = np.asarray(value)

        rows.append(
            {
                "key": key,
                "shape": str(arr.shape),
                "dtype": str(arr.dtype),
                "numeric": is_numeric_array(value),
                "ndim": arr.ndim,
                "size": arr.size,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["numeric", "size"],
        ascending=[False, False],
    )
