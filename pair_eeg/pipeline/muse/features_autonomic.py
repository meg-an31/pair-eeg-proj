"""Step 4: autonomic features — heart and breathing, the arousal side.

Unlike the EEG (features per 2 s window), these are computed per EPOCH
(~30 s), because heart-rate variability needs tens of beats to mean
anything: at 70 bpm a 30 s epoch holds ~35 beats, a 5 s window only ~6.

From the beat-to-beat (RR) intervals:
  mean_hr_bpm   average heart rate. Up = aroused (weak but robust).
  rmssd_ms      root mean square of successive RR differences — the
                standard short-term HRV measure, valid on 30 s epochs.
                DOWN = aroused (a stressed heart is steady, a calm heart
                wobbles with the breath), so stage 5 flips its sign.
  pnn50         fraction of successive-beat changes larger than 50 ms.
                Same construct as RMSSD; noisier on short epochs, logged
                for completeness but RMSSD is the one the score trusts.

From the breathing-rate trace:
  breathing_bpm mean breaths per minute. Up = aroused.

Deliberately absent: the LF/HF spectral ratio — it needs 2-5 minutes of
data and is not honest on a 30 s epoch.

Storage mirrors step 3: derived/features_autonomic.csv, one row per epoch,
with a `good` flag (an epoch with too few detected beats is kept but
flagged, features left empty).
"""

import csv
import numpy as np

MIN_BEATS = 20          # fewer detected beats than this -> unreliable epoch
FEATURE_COLUMNS = ["epoch_start_s", "epoch_end_s", "good", "n_beats",
                   "mean_hr_bpm", "rmssd_ms", "pnn50", "breathing_bpm"]


def hrv_features(rr_ms):
    """Heart-rate features from one epoch's RR intervals (milliseconds)."""
    rr = np.asarray(rr_ms, dtype=float)
    diffs = np.diff(rr)
    return {
        "n_beats": int(rr.size),
        "mean_hr_bpm": 60_000.0 / rr.mean(),
        "rmssd_ms": float(np.sqrt(np.mean(diffs ** 2))),
        "pnn50": float(np.mean(np.abs(diffs) > 50.0)),
    }


def breathing_features(rate_bpm_trace):
    """Breathing features from one epoch's breaths-per-minute trace."""
    return {"breathing_bpm": float(np.mean(rate_bpm_trace))}


def rr_beat_times_s(rr_ms):
    """Time of each beat (seconds from recording start), from RR intervals.

    Needed to slice a long recording into epochs: interval k ends at the
    cumulative sum of intervals 0..k.
    """
    return np.cumsum(np.asarray(rr_ms, dtype=float)) / 1000.0


def epoch_features(rr_ms=None, breath_t_s=None, breath_trace=None,
                   epoch_start_s=0.0, epoch_end_s=0.0):
    """All autonomic features for one time span of a recording.

    Selects the beats and breathing samples that fall inside
    [epoch_start_s, epoch_end_s) and summarises them. Returns a row dict.

    Both streams are OPTIONAL and independent. Pass None (or nothing) for a
    stream the capture setup does not provide and its features are simply
    absent from the row; the scorer already skips an index whose value is
    missing, so the remaining axes are computed from what is actually there.
    This matters because the streaming deployment has a PPG sensor but no
    respiration sensor at all, and the two used to be all-or-nothing: with no
    breathing trace the heart features were discarded too, which silently cost
    three of arousal's five voters instead of one.

    `good` means "at least one autonomic feature was measurable". Heart
    features still need MIN_BEATS beats in the span to be trustworthy; too few
    is reported as no heart features rather than as bad breathing.

    KNOWN DIVERGENCE from the ~/projects/muse original this was copied from:
    that project's `step4_demo.py` asserts a 5 s span yields `good=0`, which
    encodes the old all-or-nothing meaning. Here the same span yields `good=1`
    with the heart features absent and the breathing rate present, so that one
    check fails when the demo is replayed against this copy. The behaviour is
    intended, not a regression — but `good` is no longer a promise that all
    three features exist, which is why `score.py` reads each one only if the
    key is there.
    """
    row = {"epoch_start_s": float(epoch_start_s),
           "epoch_end_s": float(epoch_end_s)}

    rr_sel = np.asarray([], dtype=float)
    if rr_ms is not None and len(np.atleast_1d(rr_ms)):
        beat_t = rr_beat_times_s(rr_ms)
        in_epoch = (beat_t >= epoch_start_s) & (beat_t < epoch_end_s)
        rr_sel = np.asarray(rr_ms, dtype=float)[in_epoch]
    row["n_beats"] = int(rr_sel.size)

    b_sel = np.asarray([], dtype=float)
    if breath_trace is not None and breath_t_s is not None:
        breath_t_s = np.asarray(breath_t_s, dtype=float)
        b_sel = np.asarray(breath_trace, dtype=float)[
            (breath_t_s >= epoch_start_s) & (breath_t_s < epoch_end_s)]

    have_heart = rr_sel.size >= MIN_BEATS
    have_breath = b_sel.size > 0
    row["good"] = int(have_heart or have_breath)
    if have_heart:
        row.update(hrv_features(rr_sel))
    if have_breath:
        row.update(breathing_features(b_sel))
    return row


def write_features_csv(rows, path):
    """Same conventions as step 3: fixed column order, empty (not zero)
    cells for features of bad epochs."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in FEATURE_COLUMNS})


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return value
