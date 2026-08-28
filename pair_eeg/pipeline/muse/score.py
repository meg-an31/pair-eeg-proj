"""Step 5b: scoring — z-scores, sign flips, and the final valence/arousal.

The recipe (exactly as discussed while designing this):

  1. Pool: EEG indices are summarised over the epoch's good windows by
     their median; autonomic indices arrive already per-epoch.
  2. Robust z-score each index against the person's baseline:
         z = (value - baseline_median) / (1.4826 * baseline_MAD)
     The 1.4826 factor makes a MAD numerically comparable to a standard
     deviation, so these read like ordinary z-scores. Clipped to ±5 so a
     single wild index can't dominate an axis.
  3. Sign-align so that positive always means "more aroused" / "more
     pleasant". Two indices flip: RMSSD (a calm heart wobbles MORE) and
     the frown-muscle EMG (frowning = negative valence).
  3b. Muscle tension, on two timescales (see features_eeg.emg_tonic_phasic):
     the PHASIC frown response is the valence voter; the TONIC sustained
     level is a modest AROUSAL voter and is reported as its own `tension`
     column. And because muscle is broadband — it leaks down into the
     12-28 Hz beta band — a high tonic level DISCOUNTS the beta-derived
     cortical indices for that epoch (tension_guard_factor). Net effect: a
     clench is counted once, as body rather than brain.
  4. Combine: weighted mean of the aligned z-scores per axis, then squash
     with tanh(z/2) to a -1..+1 output. Mean heart rate carries half
     weight — it is the crudest arousal index, kept as a sanity anchor.
  5. Confidence: agreement across the contributing indices, 1/(1 + spread).
     Indices agreeing tightly -> near 1; contradicting each other -> low,
     which usually means an artifact somewhere. Arousal confidence is
     additionally reduced when the tension guard fires, since part of the
     evidence for that axis was deliberately discounted.

An index is silently skipped if its value is missing (an absent or
unmeasurable heart/breathing stream, no good EEG windows) or its baseline
spread is zero; the axis is then computed from whatever remains, and n_used
says how much that was.
"""

import csv
import numpy as np

from .features_eeg import good_median, emg_tonic_phasic

ROBUST_SCALE = 1.4826
Z_CLIP = 5.0
# An index's spread is never taken as smaller than 5% of its own level.
# Guards against overconfident baselines: quantities that sit on a noise
# floor (like the EMG envelope at rest) can have a tiny MAD, and then a
# few-percent floor drift between recordings would masquerade as a
# multi-sigma emotional event.
MIN_REL_SPREAD = 0.05

# The muscle indices need a much wider floor than the rest, and it is the
# same problem one step worse. At rest the frown muscle is not doing
# anything, so the envelope is not measuring muscle at all — it is measuring
# the electrode noise floor, which is very stable (MAD ~2% of its own level
# in synthetic data, and the tonic median of a 30 s epoch is steadier still).
# Scored against that, a 10% floor drift between recordings reads as a
# multi-sigma "clench". So for these indices anything below a quarter of the
# resting level counts as indistinguishable from drift. The scale comes from
# the resting TONIC level for all three, because phasic is an excursion
# measured in the same µV and its own resting value is ~0.
MUSCLE_INDICES = ("emg_tonic", "emg_phasic", "emg_envelope")
MUSCLE_REL_SPREAD = 0.25

# +1: index goes UP with the axis. -1: goes down, so its z is negated.
SIGN = {
    "beta_alpha_ratio": +1, "engagement": +1, "mean_hr_bpm": +1,
    "breathing_bpm": +1, "rmssd_ms": -1,          # arousal side
    "emg_tonic": +1,                              # tension -> arousal
    "faa": +1, "emg_phasic": -1,                  # valence side
    "emg_envelope": -1,                           # audit only, votes nowhere
}

# How much the sustained tension level counts toward arousal. Deliberately
# modest (mean-HR tier): a jaw clench should nudge arousal, never define it.
# Set to 0.0 to make tension purely diagnostic — the column and the guard
# stay, the axis stops listening.
TENSION_AROUSAL_WEIGHT = 0.5      # edit here, not at runtime: these two are
                                  # score_epoch's default arguments, bound at
                                  # import — pass them explicitly to override
                                  # a single call

# ...and how much it counts toward valence (unpleasantness). Zero by default,
# which keeps the two axes strictly independent: each physical signal votes
# on exactly one of them.
#
# Worth raising once real labelled data exists, because the timescale split
# is only a PROXY for the source split. AF7/AF8 cannot tell the frown muscle
# (corrugator — a well-validated valence measure whose TONIC level counts,
# not just its twitches) from the jaw and forehead muscles of general
# tension. At 0.0, someone frowning steadily for 30 s registers no
# unpleasantness at all and valence rests entirely on FAA, the one index
# known to reverse between individuals. At 0.5 that frown is caught, at the
# price of reading a merely clenched jaw as somewhat unpleasant. The two
# cases are indistinguishable to this hardware, so this is a real trade-off
# with no correct answer until labels arbitrate it — step8_demo.py prints
# both settings side by side so the cost is visible.
TENSION_VALENCE_WEIGHT = 0.0

AROUSAL_WEIGHTS = {"beta_alpha_ratio": 1.0, "engagement": 1.0,
                   "rmssd_ms": 1.0, "breathing_bpm": 1.0, "mean_hr_bpm": 0.5,
                   "emg_tonic": TENSION_AROUSAL_WEIGHT}
VALENCE_WEIGHTS = {"emg_phasic": 1.0, "faa": 1.0,
                   "emg_tonic": TENSION_VALENCE_WEIGHT}

# Sign is really per (index, axis). Every index but one feeds a single axis,
# so SIGN above suffices — except tonic muscle, which means MORE aroused and
# LESS pleasant at the same time. SIGN holds its arousal-facing sign (so the
# z_ audit columns and the `tension` column all read "positive = tenser"),
# and this override flips it on the way into valence.
VALENCE_SIGN_OVERRIDE = {"emg_tonic": -1}

# The guard: indices built on beta power, which muscle contaminates from
# above. FAA is alpha-based and is left alone.
GUARDED_INDICES = ("beta_alpha_ratio", "engagement")
GUARD_Z_START = 1.0    # tonic-EMG z at which distrust begins
GUARD_Z_FULL = 4.0     # ...and at which it is maximal
GUARD_FLOOR = 0.25     # never discount a cortical index below this fraction

SCORE_COLUMNS = ["t_s", "valence", "arousal", "tension",
                 "confidence_valence", "confidence_arousal",
                 "n_eeg_windows", "tension_guard",
                 "z_beta_alpha_ratio", "z_engagement", "z_rmssd_ms",
                 "z_mean_hr_bpm", "z_breathing_bpm", "z_faa",
                 "z_emg_tonic", "z_emg_phasic", "z_emg_envelope"]


def robust_z(value, stats, min_spread=0.0):
    """Robust z-score of one value against its baseline {median, mad}.

    min_spread is an absolute floor on the spread, for indices whose resting
    MAD is known to understate how much they can drift (see MUSCLE_REL_SPREAD).
    Returns None when it cannot be computed honestly.
    """
    if value is None or not np.isfinite(value):
        return None
    if not np.isfinite(stats["median"]):
        return None
    spread = max(stats["mad"], MIN_REL_SPREAD * abs(stats["median"]), min_spread)
    if spread <= 0:
        return None
    z = (value - stats["median"]) / (ROBUST_SCALE * spread)
    return float(np.clip(z, -Z_CLIP, Z_CLIP))


def tension_guard_factor(z_tonic):
    """How much of their vote the beta-derived cortical indices keep, given
    how tense this epoch is. 1.0 = fully trusted, GUARD_FLOOR = maximally
    discounted, ramped linearly in between.

    Facial muscle activity is broadband — it does not begin at 55 Hz where we
    measure it, it extends downward into the 12-28 Hz beta band that
    beta_alpha_ratio and engagement are built from. A gentle sustained clench
    is far too small to trip the +-100 uV artifact rejection, so it survives
    into the "clean" cortical path and inflates beta: tension masquerading as
    cortical arousal. When the sustained EMG level is well above this person's
    rest, some of the measured beta is jaw, and those indices earn less say.

    Never zero: the epoch may be genuinely aroused AND tense, and discarding
    the cortical evidence entirely would throw that away too.
    """
    if z_tonic is None or not np.isfinite(z_tonic) or z_tonic <= GUARD_Z_START:
        return 1.0
    frac = min((z_tonic - GUARD_Z_START) / (GUARD_Z_FULL - GUARD_Z_START), 1.0)
    return float(1.0 - frac * (1.0 - GUARD_FLOOR))


def _axis(aligned_z, weights, sign_override=None):
    """One axis from sign-aligned z-scores: weighted mean -> tanh squash,
    plus an agreement-based confidence.

    sign_override flips individual indices for this axis (see
    VALENCE_SIGN_OVERRIDE) — needed only where one index feeds both axes in
    opposite directions.
    """
    sign_override = sign_override or {}
    used = {ix: z * sign_override.get(ix, 1) for ix, z in aligned_z.items()
            if ix in weights and z is not None}
    if not used:
        return {"score": float("nan"), "confidence": 0.0, "n_used": 0}
    w = np.array([weights[ix] for ix in used])
    z = np.array(list(used.values()))
    combined = float(np.sum(w * z) / np.sum(w))
    spread = float(np.std(z)) if len(z) > 1 else 1.0   # single index: mediocre trust
    return {"score": float(np.tanh(combined / 2.0)),
            "confidence": 1.0 / (1.0 + spread),
            "n_used": len(used)}


def _with_tension(weights, tension_weight):
    """Weights for one axis with the tonic-muscle weight substituted in.
    A zero weight is dropped rather than kept at 0, so a disabled index does
    not still get a vote in the agreement-based confidence."""
    out = {ix: w for ix, w in weights.items() if ix != "emg_tonic" and w > 0}
    if tension_weight and tension_weight > 0:
        out["emg_tonic"] = float(tension_weight)
    return out


def score_epoch(eeg_rows, autonomic_row, baseline, t_s=None,
                tension_weight=TENSION_AROUSAL_WEIGHT,
                tension_valence_weight=TENSION_VALENCE_WEIGHT, guard=True):
    """Valence/arousal for one epoch.

    eeg_rows        step-3 feature rows whose windows fall inside the epoch
    autonomic_row   step-4 row for the same span (or None / good=0)
    baseline        dict from compute_baseline / load_baseline
    tension_weight  how much sustained muscle tension counts toward arousal
    tension_valence_weight
                    ...and toward valence (as unpleasantness). Both 0.0
                    makes the `tension` column purely diagnostic.
    guard           whether a tense epoch discounts the beta-derived
                    cortical indices (see tension_guard_factor)

    Returns a dict shaped like one row of scores.csv.
    """
    stats = baseline["indices"]

    # raw index values for this epoch
    values = {ix: good_median(eeg_rows, ix)
              for ix in ("beta_alpha_ratio", "engagement", "faa", "emg_envelope")}
    values.update(emg_tonic_phasic(eeg_rows))     # epoch-level muscle split
    if autonomic_row and autonomic_row.get("good"):
        # Take whichever autonomic features are actually there. The heart and
        # breathing streams are independent and either may be absent — a rig
        # with a pulse sensor and no respiration belt, or an epoch holding too
        # few beats to trust — so `good` means "at least one is measurable",
        # not "all three are". Indexing all three unconditionally raised
        # KeyError on exactly that case; skipping an absent one is what this
        # function already does for every EEG index a few lines below.
        for ix in ("mean_hr_bpm", "rmssd_ms", "breathing_bpm"):
            if ix in autonomic_row:
                values[ix] = autonomic_row[ix]

    # the muscle indices share one absolute spread floor, set by the resting
    # tonic level (see MUSCLE_REL_SPREAD)
    tonic_rest = stats.get("emg_tonic", {}).get("median", float("nan"))
    muscle_floor = (MUSCLE_REL_SPREAD * abs(tonic_rest)
                    if np.isfinite(tonic_rest) else 0.0)

    # robust z, then sign-align (positive = more aroused / more pleasant,
    # except emg_tonic which is aligned to arousal and flipped for valence)
    aligned = {}
    for ix, value in values.items():
        floor = muscle_floor if ix in MUSCLE_INDICES else 0.0
        z = robust_z(value, stats[ix], floor) if ix in stats else None
        aligned[ix] = None if z is None else SIGN[ix] * z

    # tension: how far above rest the sustained muscle level sits. Positive
    # = tenser. Reported on the same -1..+1 scale as the axes so it can be
    # read alongside them, and used to discount the beta-based indices.
    z_tonic = aligned.get("emg_tonic")
    factor = tension_guard_factor(z_tonic) if guard else 1.0
    arousal_weights = _with_tension(AROUSAL_WEIGHTS, tension_weight)
    valence_weights = _with_tension(VALENCE_WEIGHTS, tension_valence_weight)
    for ix in GUARDED_INDICES:
        if ix in arousal_weights:
            arousal_weights[ix] *= factor

    arousal = _axis(aligned, arousal_weights)
    valence = _axis(aligned, valence_weights, VALENCE_SIGN_OVERRIDE)
    # some of the arousal evidence was deliberately discounted; a full guard
    # halves the confidence in what is left.
    arousal["confidence"] *= 0.5 + 0.5 * factor

    row = {"t_s": t_s, "valence": valence["score"], "arousal": arousal["score"],
           "tension": (float(np.tanh(z_tonic / 2.0)) if z_tonic is not None
                       else float("nan")),
           "confidence_valence": valence["confidence"],
           "confidence_arousal": arousal["confidence"],
           "n_eeg_windows": sum(r["good"] for r in eeg_rows),
           "tension_guard": factor}
    for ix, z in aligned.items():
        row[f"z_{ix}"] = z          # stored sign-ALIGNED, the audit trail
    return row


def write_scores_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_COLUMNS, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in SCORE_COLUMNS})


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return value
