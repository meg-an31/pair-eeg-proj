"""Step 3: EEG features — from clean windows to per-window index values.

Takes the output of step 2 (preprocess_eeg) and computes, for every GOOD
window, the established indices chosen for this project:

  arousal side (cortical path):
    beta_alpha_ratio  beta / alpha power, frontal channels. Up = aroused.
    engagement        beta / (alpha + theta), all channels. Up = engaged.
  valence side:
    faa               frontal alpha asymmetry ln(alpha AF8) - ln(alpha AF7).
                      Up = left hemisphere more active = approach motivation.
    emg_envelope      RMS amplitude (µV) of the EMG path on the frontal
                      channels — frown-muscle activity. Up = NEGATIVE valence.
                      Split at epoch level into tonic/phasic components by
                      emg_tonic_phasic() below; see that function.

Raw ingredients (per-channel band powers) are stored alongside, so nothing
needs recomputing when formulas are tweaked later.

Values here are RAW index values in their own units. Turning them into
comparable scores (z-scoring against a rest baseline, sign flips, averaging
into valence/arousal) is deliberately left to stage 5.

Storage: one CSV per session, one row per window. Bad windows keep their
row (good=0, feature cells empty) so row n is always the window starting
at second n and the rejection rate stays visible.
"""

import csv
import numpy as np
from scipy.signal import welch

BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 12.0), "beta": (12.0, 28.0)}

CHANNELS = ("TP9", "AF7", "AF8", "TP10")        # must match capture order
FRONTAL = ("AF7", "AF8")

# column order of the on-disk table
FEATURE_COLUMNS = [
    "win_start_s", "win_end_s", "good",
    "theta_tp9", "theta_af7", "theta_af8", "theta_tp10",
    "alpha_tp9", "alpha_af7", "alpha_af8", "alpha_tp10",
    "beta_tp9", "beta_af7", "beta_af8", "beta_tp10",
    "beta_alpha_ratio", "engagement", "faa", "emg_envelope",
]


def band_powers(window_data, fs):
    """Mean power in each band for each channel of one window.

    Welch's method with 1-second segments -> 1 Hz frequency resolution,
    enough to separate theta/alpha/beta. Returns {band: array of 4}.
    """
    freqs, psd = welch(window_data, fs=fs, nperseg=int(fs), axis=-1)
    out = {}
    for band, (lo, hi) in BANDS.items():
        sel = (freqs >= lo) & (freqs < hi)
        out[band] = psd[:, sel].mean(axis=-1)
    return out


def _window_features(cortical_win, emg_win, fs):
    """The four indices + raw band powers for one good window."""
    powers = band_powers(cortical_win, fs)
    i_af7, i_af8 = CHANNELS.index("AF7"), CHANNELS.index("AF8")

    alpha_frontal = powers["alpha"][[i_af7, i_af8]].mean()
    beta_frontal = powers["beta"][[i_af7, i_af8]].mean()

    feats = {}
    for band in BANDS:
        for i, ch in enumerate(CHANNELS):
            feats[f"{band}_{ch.lower()}"] = powers[band][i]

    feats["beta_alpha_ratio"] = beta_frontal / alpha_frontal
    feats["engagement"] = powers["beta"].mean() / (
        powers["alpha"].mean() + powers["theta"].mean())
    feats["faa"] = np.log(powers["alpha"][i_af8]) - np.log(powers["alpha"][i_af7])
    # RMS of the >55 Hz path on the frontal channels = frown-muscle envelope
    feats["emg_envelope"] = float(np.sqrt(np.mean(emg_win[[i_af7, i_af8]] ** 2)))
    return feats


def eeg_features(pre):
    """Feature rows for every window of a preprocessed recording.

    `pre` is the dict returned by preprocess_eeg. Returns a list of dicts,
    one per window in time order; bad windows carry only timing + good=0.
    """
    fs = pre["fs"]
    rows = []
    for (start, stop), good in zip(pre["windows"], pre["good"]):
        row = {"win_start_s": start / fs, "win_end_s": stop / fs,
               "good": int(good)}
        if good:
            row.update(_window_features(pre["cortical"][:, start:stop],
                                        pre["emg"][:, start:stop], fs))
        rows.append(row)
    return rows


def write_features_csv(rows, path):
    """Write feature rows to CSV in the standard column order.

    Missing cells (features of bad windows) are written empty, not zero —
    zero would look like a measurement.
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in FEATURE_COLUMNS})


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def good_median(rows, feature):
    """Median of one feature over the good windows — the epoch summary."""
    values = [r[feature] for r in rows if r["good"]]
    return float(np.median(values)) if values else float("nan")


# --------------------------------------------------------------------------
# Epoch-level muscle features: the tonic/phasic split
# --------------------------------------------------------------------------
#
# The frown-muscle envelope carries two different physiological stories on
# two different timescales, and they belong in different places:
#
#   phasic  short-lived elevation above the epoch's own slow level — the
#           frown that arrives WITH a stimulus. A reaction.   -> VALENCE
#   tonic   the sustained level itself — jaw or forehead held tight for tens
#           of seconds. A state.        -> AROUSAL, and reported as `tension`
#
# Why bother splitting: sustained tension and a momentary frown are the same
# number in `emg_envelope`, so a single clench would otherwise vote on both
# axes at once — double-counting one physical event and weakening the axes'
# independence. Reactions and states are genuinely different physiology.
#
# Both are statistics OF AN EPOCH, not of a window (a single 2 s window has
# no notion of its own baseline), so unlike the indices above they are
# z-scored against the distribution of the same 30 s statistic over the rest
# recording — see EEG_EPOCH_INDICES in baseline.py.

PHASIC_PERCENTILE = 90.0    # "how high does the envelope briefly get"
MIN_PHASIC_WINDOWS = 5      # below this a percentile is meaningless


def emg_tonic_phasic(rows):
    """Tonic level and phasic elevation of the frown-muscle envelope over one
    epoch's feature rows.

    Returns {"emg_tonic": µV, "emg_phasic": µV}. Tonic is the median of the
    good windows (robust to brief spikes by construction); phasic is how far
    the 90th percentile rises above that median, so it measures short-lived
    excursions and is ~0 for a steady clench however hard. Either value is
    nan when the epoch has too few clean windows to support it, and nan
    propagates to "index skipped" in the scorer.
    """
    values = [r["emg_envelope"] for r in rows if r["good"]]
    if not values:
        return {"emg_tonic": float("nan"), "emg_phasic": float("nan")}
    v = np.asarray(values, dtype=float)
    tonic = float(np.median(v))
    if v.size < MIN_PHASIC_WINDOWS:
        return {"emg_tonic": tonic, "emg_phasic": float("nan")}
    return {"emg_tonic": tonic,
            "emg_phasic": float(np.percentile(v, PHASIC_PERCENTILE)) - tonic}
