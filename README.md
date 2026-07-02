# ML-project-Arthur-Feld

# ML-project-Arthur-Feld

Predicting the **annoyance score** of aircraft noise from spectrogram data, starting
from a minimal linear-regression baseline and progressively building toward a richer,
feature-based model.

The input data are spectrograms (one `.mat` file per recorded signal). The target is a
subjective annoyance score for each signal.

---

## Objective

Build a model that maps spectrogram-derived features to an annoyance score. The project
is intentionally staged: begin with the simplest possible predictor (a single feature,
linear model) and only add complexity once the baseline is understood and evaluated.

---

## Repository structure

```
ML-project-Arthur-Feld/
├── data/
│   └── raw/                          # the ~77 .mat spectrogram files (one per signal)
├── features/
│   └── mean_frequency_features.csv   # extracted features (output)
├── scripts/                          # runnable entry points (called from terminal)
│   ├── extract_mean_frequencies.py   # loops all .mat files → writes feature CSV
│   └── inspect_mat_file.py           # inspects the variables inside one .mat file
├── src/                              # reusable core functions
│   └── mat_spectrogram_features.py   # load .mat, find freq/spectrogram, dB→linear, mean freq
├── tests/
│   └── test_mean_frequency.py        # sanity checks for the mean-frequency math
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

## Usage

**Inspect a single `.mat` file** (see the variables and their shapes):

```bash
py scripts/inspect_mat_file.py "data/raw/SignalNum105_B777-300_Flyover.mat"
```

**Extract mean-frequency features from every file** into the feature CSV:

```bash
py scripts/extract_mean_frequencies.py data/raw \
    --freq-key freq --spectrogram-key psdx_dB --values-are-db true
```

Each `.mat` file contains a spectrogram `psdx_dB` (frequency × time, in dB), a frequency
vector `freq`, and a time vector `t`.

---

## Project phases

### Phase 1 — Understanding the tools and the problem

Going through the data provided by Professor Redonnet and building the necessary
background:
- how music/audio generation with AI works, and how a signal is decomposed into
  frequency parameters;
- how to properly apply machine learning in Python;
- how to work with spectrograms as model inputs.

### Phase 2 — Developing the MVP

A simple **linear regression** model trained on the available samples (~77 spectrograms),
targeted for the end of week 1.

**MVP idea:** compute the mean frequency of each spectrogram between `t = 0` and
`t = tmax`, and use that single value as the feature `X`. The model does not attempt to
identify any patterns — it predicts the annoyance score directly from the mean frequency
alone.

**Next model:** decompose each spectrogram into `d` features (each feature being a mean
over `t = 0` to `t = tmax`), so that the input becomes a vector `X ∈ ℝ^d`. Fit a new
linear model on this richer input:

$$h(x) = \sum_{i=1}^{d} \theta_i \, x_i$$

Then identify which features can be neglected in order to simplify the model.

### Phase 3 — Improving the model

Extend the MVP into a more detailed and realistic model.

---

## Evaluation

For each model:
- split the samples into **80% training / 20% test**;
- plot the prediction graph (predicted vs. actual annoyance score);
- compute the **RMSE**:

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( h(x_i) - y_i \right)^2}$$