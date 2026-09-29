# ML-project-Arthur-Feld

Predicting the **annoyance score** of aircraft noise from spectrogram data, starting
from a minimal linear-regression baseline and progressively building toward a richer,
feature-based model.

The input data are spectrograms (one `.mat` file per recorded signal). The target is a
subjective annoyance score for each signal, measured in a psychoacoustic listening
study.

---

## Objective

Build a model that maps spectrogram-derived features to an annoyance score. The project
is intentionally staged: begin with the simplest possible predictor (a single feature,
linear model) and only add complexity once the baseline is understood and evaluated.

---

## Data

- **Inputs:** ~77 aircraft-noise signals in `data/raw/`, one `.mat` file per signal.
  Each file contains a spectrogram `psdx_dB` (frequency x time, in dB), a frequency
  vector `freq` and a time vector `t`. Every signal was recorded in one of three
  flight phases (Approach, Flyover, Recede), encoded in its filename.
- **Labels:** `data/labels/PsychoResults_All.xlsx`, the results of a listening study
  in which 55 participants rated each (signal, phase) stimulus on a 0 to 5 annoyance
  scale. The label of a stimulus is the mean rating across the participants; stimuli
  presented twice (test-retest) are averaged together.
- The two sides are joined on (SignalNum, Phase), both parsed from the filenames.

---

## Repository structure

```
ML-project-Arthur-Feld/
├── data/
│   ├── raw/                          # the ~77 .mat spectrogram files (model inputs)
│   └── labels/                       # PsychoResults_All.xlsx (targets)
├── features/
│   ├── mean_frequency_features.csv   # one extracted feature row per .mat file
│   └── dataset.csv                   # features joined to labels (the modeling table)
├── reports/
│   ├── split_8020/  split_7030/  split_5545/   # single-split experiments
│   └── kfold/                        # holdout + 5-fold results (the reported ones)
├── scripts/                          # runnable entry points (thin CLI wrappers)
│   ├── inspect_mat_file.py           # show the variables inside one .mat file
│   ├── extract_mean_frequencies.py   # loop over all .mat files : write the feature CSV
│   ├── build_dataset.py              # join the features to the annoyance labels
│   ├── train_model.py                # fit and evaluate on one train/test split
│   ├── train_model_kfold.py          # holdout + 5-fold procedure (reported results)
│   └── plot_actual_vs_predicted.py   # real vs predicted figure on the test set
├── src/                              # reusable core logic, imported by the scripts
│   ├── mat_spectrogram_features.py   # load .mat, find variables, dB conversion, centroid
│   ├── dataset.py                    # label aggregation and the feature/label join
│   └── model.py                      # fitting, metrics, plots, cross-validation
├── tests/                            # verification on inputs with known answers
│   ├── test_mean_frequency.py        # 6 hand-computed centroid checks
│   ├── test_sinusoid_recovery.py     # 23 checks from real waveforms through the FFT
│   ├── plot_test_signals.py          # figures illustrating the synthetic test signals
│   └── figures/                      # the figures the script above produces
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

```bash
py -m venv venv
source venv/Scripts/activate     # Windows (Git Bash)
py -m pip install -r requirements.txt
```

---

## Usage : the full pipeline

Every result in `reports/` can be regenerated with the commands below, in order.
All splits use a fixed random seed, so the numbers reproduce exactly.

**1. Extract the mean-frequency feature from every file:**

```bash
py scripts/extract_mean_frequencies.py data/raw \
    --freq-key freq --spectrogram-key psdx_dB --values-are-db true
```

**2. Join the features to the annoyance labels:**

```bash
py scripts/build_dataset.py
```

**3. Train and evaluate on one 80/20 split:**

```bash
py scripts/train_model.py --test-size 0.20 --output-dir reports/split_8020
```

**4. Run the holdout + 5-fold procedure (the reported results):**

```bash
py scripts/train_model_kfold.py
```

**5. Plot real vs predicted annoyance on the held-out test set:**

```bash
py scripts/plot_actual_vs_predicted.py --test-size 0.20 --output-dir reports/split_8020
```

To look inside one `.mat` file (used at the start of the project to pin the
variable keys):

```bash
py scripts/inspect_mat_file.py data/raw/SignalNum107_A350-941_Approach.mat
```

---

## Verification

Every non-trivial computation is checked against inputs with known answers before
being trusted on the real data:

```bash
py tests/test_mean_frequency.py      # 6 checks on hand-made power matrices
py tests/test_sinusoid_recovery.py   # 23 checks from waveform to recovered frequency
```

`tests/plot_test_signals.py` draws the synthetic signals used by the second file;
its figures are visual companions to the checks, saved in `tests/figures/`.

---

## Evaluation

For each model:
- split the samples into **80% training / 20% test**;
- plot the prediction graph (predicted vs. actual annoyance score);
- compute the **RMSE**:

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( h(x_i) - y_i \right)^2}$$

Because single-split results vary with the split (see `reports/split_*`), the reported
results use a stricter procedure (`train_model_kfold.py`): 20% of the signals are held
out, the remaining 80% are split into 5 folds, the fold with the lowest validation
RMSE is selected, and that model is scored once on the untouched holdout.

**Reported MVP results (holdout): RMSE = 0.3498, R² = 0.6075.**

---

## Project phases

### Phase 1 : understanding the tools and the problem

Going through the data provided by Professor Redonnet and building the necessary
background: how a signal is described in the frequency domain, how spectrograms are
used as model inputs, and how to apply machine learning to them in Python.

### Phase 2 : developing the MVP

A simple **linear regression** on a single feature: the power-weighted mean frequency
(spectral centroid) of each spectrogram. The dB values are converted to linear power
(P = 10^(dB/10)) before weighting, because dB is logarithmic and cannot be averaged
directly. The model predicts the annoyance score from that one value.

The natural extension is a richer input: decompose each spectrogram into `d` features,
so the input becomes a vector `X ∈ ℝ^d`, and fit

$$h(x) = \sum_{i=1}^{d} \theta_i \, x_i$$

### Phase 3 : improving the model

Investigation of a richer, dynamics-based feature set (Dynamic Mode Decomposition),
carried out on the `dmd-features` branch and merged into `main` once finalized.