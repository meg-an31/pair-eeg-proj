"""Step 5a: the baseline — what "normal, at rest" looks like for this person.

Raw index values (a beta/alpha ratio of 0.4, an RMSSD of 48 ms) mean nothing
by themselves; they only become interpretable relative to the same person's
resting distribution. This module builds that distribution from a ~2 minute
eyes-open rest recording:

  - EEG indices: one value per good 2 s window (~80+ values from 2 min)
  - epoch-level EEG indices (tonic/phasic muscle): one value per sliding
    30 s span stepping 5 s — these are statistics OF a 30 s epoch, so they
    must be compared against the spread of the same 30 s statistic at rest,
    not against the much wider per-window spread
  - autonomic indices: one value per sliding 30 s span stepping 5 s
    (~19 values from 2 min)

Each index is summarised by its MEDIAN and MAD (median absolute deviation)
rather than mean/std — with only dozens of values from dry electrodes, a
couple of missed artifacts would wreck a mean, but barely move a median.

The result is saved as baseline.json so sessions can be (re-)scored against
any baseline without recomputing features, and so baselines can be compared
across days as a sanity check.

Quality checks run automatically and come back as human-readable warnings —
a baseline can be quietly garbage (bad fit, person not actually at rest),
and then every score in the session inherits it.
"""

import json
import numpy as np

from .features_eeg import eeg_features, emg_tonic_phasic
from .features_autonomic import epoch_features

EEG_INDICES = ["beta_alpha_ratio", "engagement", "faa", "emg_envelope"]
EEG_EPOCH_INDICES = ["emg_tonic", "emg_phasic"]
AUTONOMIC_INDICES = ["mean_hr_bpm", "rmssd_ms", "breathing_bpm"]

EPOCH_S = 30.0     # autonomic span length used for the baseline distribution
STEP_S = 5.0


def _median_mad(values):
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return {"median": med, "mad": mad, "n": int(values.size)}


def compute_baseline(pre_rest, rr_ms=None, breath_t=None, breath_trace=None):
    """Baseline statistics from a preprocessed rest recording.

    pre_rest: preprocess_eeg() output for the rest EEG.
    Returns a dict ready to be saved as baseline.json.

    The autonomic streams are OPTIONAL, matching epoch_features: a rest
    recording that carries EEG alone produces an EEG-only baseline, and the
    scorer then computes its axes from the indices that do have statistics. An
    index with no baseline entry is skipped rather than scored against zero,
    which would read as a large deviation from a reference that never existed.
    """
    rows = eeg_features(pre_rest)
    good_rows = [r for r in rows if r["good"]]
    duration_s = rows[-1]["win_end_s"] if rows else 0.0

    indices = {}
    for ix in EEG_INDICES:
        indices[ix] = _median_mad([r[ix] for r in good_rows])

    # epoch-level EEG indices: same sliding 30 s / 5 s spans, but over the
    # window rows we already have (no re-filtering).
    eeg_epoch_rows = []
    start = 0.0
    while start + EPOCH_S <= duration_s:
        span = [r for r in rows if r["win_start_s"] >= start - 1e-9
                and r["win_end_s"] <= start + EPOCH_S + 1e-9]
        vals = emg_tonic_phasic(span)
        if all(np.isfinite(v) for v in vals.values()):
            eeg_epoch_rows.append(vals)
        start += STEP_S
    for ix in EEG_EPOCH_INDICES:
        values = [r[ix] for r in eeg_epoch_rows]
        indices[ix] = _median_mad(values) if values else {"median": float("nan"),
                                                         "mad": 0.0, "n": 0}

    # autonomic distribution from sliding spans over the rest recording
    auto_rows = []
    start = 0.0
    while start + EPOCH_S <= duration_s and (rr_ms is not None
                                             or breath_trace is not None):
        row = epoch_features(rr_ms, breath_t, breath_trace, start, start + EPOCH_S)
        if row["good"]:
            auto_rows.append(row)
        start += STEP_S
    for ix in AUTONOMIC_INDICES:
        values = [r[ix] for r in auto_rows if ix in r]
        indices[ix] = _median_mad(values) if values else {"median": float("nan"),
                                                          "mad": 0.0, "n": 0}

    baseline = {
        "rest_duration_s": float(duration_s),
        "n_windows": len(rows),
        "n_good_windows": len(good_rows),
        "n_eeg_epochs": len(eeg_epoch_rows),
        "n_autonomic_epochs": len(auto_rows),
        "indices": indices,
        "warnings": _quality_warnings(rows, good_rows, indices),
    }
    return baseline


def _quality_warnings(rows, good_rows, indices):
    """The automatic trust checks. Empty list = nothing suspicious."""
    warnings = []

    if rows and len(good_rows) / len(rows) < 0.5:
        warnings.append(
            f"only {len(good_rows)}/{len(rows)} windows survived artifact "
            "rejection — bad electrode fit? Re-seat the headband and re-record.")

    if len(good_rows) < 30:
        warnings.append(
            f"only {len(good_rows)} clean windows — baseline statistics will "
            "be shaky; record a longer rest period.")

    # stationarity: was the person actually settled? Compare each EEG index
    # between the first and second half of the rest period.
    half_t = (rows[-1]["win_end_s"] / 2.0) if rows else 0.0
    for ix in EEG_INDICES:
        first = [r[ix] for r in good_rows if r["win_start_s"] < half_t]
        second = [r[ix] for r in good_rows if r["win_start_s"] >= half_t]
        if not first or not second:
            continue
        mad = indices[ix]["mad"]
        if mad > 0 and abs(np.median(first) - np.median(second)) > 2.0 * mad:
            warnings.append(
                f"{ix} drifted between the first and second half of the rest "
                "period — person may not have been settled; consider "
                "discarding the first 30 s.")

    for ix, stats in indices.items():
        if stats["n"] > 0 and stats["mad"] == 0.0:
            warnings.append(
                f"{ix} has zero spread in the baseline — its z-scores would "
                "explode; this index will be skipped when scoring.")

    return warnings


def save_baseline(baseline, path):
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)


def load_baseline(path):
    with open(path) as f:
        return json.load(f)
