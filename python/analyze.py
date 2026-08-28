#!/usr/bin/env python3
"""
Analyse a Muse 2 recording produced by capture.py (or synth.py).

  python analyze.py data/20260828-112416_synthetic

Does: filtering -> artifact rejection -> epoching by condition -> Welch PSD ->
band powers -> eyes-closed vs eyes-open statistics -> figures + summary files.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal, stats

import matplotlib
matplotlib.use("Agg")           # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt

BANDS = {
    "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
    "beta": (13, 30), "gamma": (30, 45),
}
COND = {1: "eyes_open", 2: "eyes_closed"}


# ------------------------------------------------------------------- io ------
def load(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    raw = np.genfromtxt(run_dir / "eeg.csv", delimiter=",", names=True)
    names = meta["eeg"]["channels"]
    t = raw["time_s"]
    eeg = np.vstack([raw[c] for c in names])
    marker = raw["marker"] if "marker" in raw.dtype.names else np.zeros_like(t)
    return meta, t, eeg, marker, names


# ---------------------------------------------------------- preprocessing ----
def preprocess(eeg, fs, line_hz):
    """Notch out mains, bandpass 1-45 Hz, zero-phase (filtfilt)."""
    out = signal.detrend(eeg, axis=1)
    for f0 in (line_hz, line_hz * 2):
        if f0 < fs / 2 - 1:
            b, a = signal.iirnotch(f0, Q=30, fs=fs)
            out = signal.filtfilt(b, a, out, axis=1)
    sos = signal.butter(4, [1.0, 45.0], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, out, axis=1)


def label_samples(t, marker, meta):
    """Map every sample to a condition using markers, falling back to meta timeline."""
    labels = np.full(len(t), "", dtype=object)
    events = [(t[i], int(m)) for i, m in enumerate(marker) if m in COND]
    if not events:
        events = [(e["t_s"], e["marker"]) for e in meta.get("timeline", [])
                  if e["marker"] in COND]
    if not events:
        labels[:] = "all"
        return labels
    events.sort()
    for i, (t_on, code) in enumerate(events):
        t_off = events[i + 1][0] if i + 1 < len(events) else t[-1] + 1
        labels[(t >= t_on) & (t < t_off)] = COND[code]
    return labels


def epoch(eeg, t, labels, fs, win_s=2.0, overlap=0.5, reject_uv=150.0, edge_s=1.0):
    """Split into fixed windows. Drops windows that span conditions, sit on a
    block edge (transition smear), or exceed the amplitude threshold."""
    step = int(win_s * fs * (1 - overlap))
    n_win = int(win_s * fs)
    epochs, conds, kept, dropped, peaks = [], [], 0, 0, []

    # samples where the condition label changes = block boundaries
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    edges = t[change] if change.size else np.array([])

    for s in range(0, eeg.shape[1] - n_win, step):
        e = s + n_win
        seg_labels = labels[s:e]
        if seg_labels[0] == "" or not np.all(seg_labels == seg_labels[0]):
            continue
        if edges.size and np.any(np.abs(edges - t[s]) < edge_s):
            continue
        seg = eeg[:, s:e]
        pk = float(np.max(np.abs(seg - seg.mean(axis=1, keepdims=True))))
        peaks.append(pk)
        if pk > reject_uv:
            dropped += 1
            continue
        epochs.append(seg)
        conds.append(seg_labels[0])
        kept += 1
    return np.array(epochs), np.array(conds), kept, dropped, np.array(peaks)


# ------------------------------------------------------------- spectral ------
def psd(epochs, fs):
    """Welch PSD per epoch per channel -> (n_epochs, n_ch, n_freqs)."""
    f, p = signal.welch(epochs, fs=fs, nperseg=min(epochs.shape[-1], int(fs)),
                        axis=-1, detrend="constant")
    return f, p


def band_power(f, p, band, relative=False):
    lo, hi = band
    idx = (f >= lo) & (f < hi)
    bp = np.trapezoid(p[..., idx], f[idx], axis=-1)
    if relative:
        tot = np.trapezoid(p[..., (f >= 1) & (f < 45)], f[(f >= 1) & (f < 45)], axis=-1)
        return bp / np.where(tot == 0, np.nan, tot)
    return bp


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else float("nan")


# ----------------------------------------------------------------- ppg -------
PPG_BAND = (0.7, 3.5)          # 42-210 bpm
PPG_SNR_MIN = 8.0              # below this the cardiac band is indistinguishable
PPG_SNR_HRV = 50.0             # HRV needs a much cleaner trace than a rate does


def ppg_pulsatile(x, fs):
    """Dominant frequency in the cardiac band, and how far it stands above that
    band's own background. Returns (hz, snr) or None.

    Frequency-domain because it does not care about peak shape: a rate estimate
    survives baseline drift, motion bursts and a half-detached sensor, all of
    which wreck peak counting.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20 * fs:
        return None
    nper = int(min(x.size, 32 * fs))            # >=32 s -> ~1.9 bpm resolution
    f, pxx = signal.welch(signal.detrend(x), fs=fs, nperseg=nper)
    band = (f >= PPG_BAND[0]) & (f <= PPG_BAND[1])
    if band.sum() < 3:
        return None
    fb, pb = f[band], pxx[band]
    k = int(np.argmax(pb))
    hz = float(fb[k])
    # parabolic interpolation on the log spectrum for sub-bin accuracy
    if 0 < k < len(pb) - 1:
        a, b, c = np.log(pb[k-1] + 1e-30), np.log(pb[k] + 1e-30), np.log(pb[k+1] + 1e-30)
        denom = a - 2*b + c
        if denom != 0:
            hz = float(fb[k] + 0.5 * (a - c) / denom * (fb[1] - fb[0]))
    med = float(np.median(pb))
    return hz, (float(pb[k]) / med if med > 0 else 0.0)


def ppg_channels(data):
    return [n for n in data.dtype.names if n.upper().startswith("PPG")]


def heart_rate(run_dir, meta):
    """Heart rate from PPG, with the channel chosen by signal quality.

    The rate comes from the spectrum; peak detection is used only for HRV and
    only when the trace is clean enough to justify it. An earlier version read a
    hardcoded channel and counted peaks with a std-based prominence, which on a
    poor trace silently found a fraction of the beats and reported a confident
    but meaningless number.
    """
    path = Path(run_dir) / "ppg.csv"
    if not path.exists() or meta.get("ppg", {}).get("n_samples", 0) < 100:
        return None
    d = np.genfromtxt(path, delimiter=",", names=True)
    fs = meta["ppg"]["fs"]

    scored = []
    for name in ppg_channels(d):
        r = ppg_pulsatile(d[name], fs)
        if r:
            scored.append((r[1], r[0], name))
    if not scored:
        return None
    snr, hz, chan = max(scored)
    bpm = 60.0 * hz

    out = {"bpm_mean": round(bpm, 1), "channel": chan, "snr": round(snr, 1),
           "fs": fs, "quality": "ok"}

    if snr < PPG_SNR_MIN:
        out.update(quality="unusable", bpm_mean=None,
                   note="no cardiac rhythm above the noise floor on any channel")
        return out

    # --- beat detection, for HRV and as a cross-check on the spectral rate ----
    x = np.asarray(d[chan], dtype=float)
    x = x[np.isfinite(x)]
    sos = signal.butter(3, PPG_BAND, btype="bandpass", fs=fs, output="sos")
    xf = signal.sosfiltfilt(sos, signal.detrend(x))
    mad = float(np.median(np.abs(xf - np.median(xf))))
    prom = max(1.4826 * mad * 0.6, 1e-9)        # robust to artifact spikes
    peaks, _ = signal.find_peaks(xf, distance=int(0.5 * (60.0 / bpm) * fs),
                                 prominence=prom)
    secs = x.size / fs
    expected = secs * bpm / 60.0
    out["n_beats"] = int(len(peaks))
    out["expected_beats"] = int(round(expected))
    out["beat_coverage"] = round(len(peaks) / expected, 2) if expected else None

    ibi = np.diff(peaks) / fs
    if ibi.size:
        centre = 60.0 / bpm
        ibi = ibi[(ibi > 0.5 * centre) & (ibi < 1.6 * centre)]
    if ibi.size >= 4:
        out["bpm_peaks"] = round(float(60.0 / np.mean(ibi)), 1)
        out["bpm_sd"] = round(float(np.std(60.0 / ibi)), 1)

    good_coverage = out["beat_coverage"] is not None and 0.85 <= out["beat_coverage"] <= 1.15
    if ibi.size >= 5 and snr >= PPG_SNR_HRV and good_coverage:
        out["rmssd_ms"] = round(float(np.sqrt(np.mean(np.diff(ibi) ** 2)) * 1000), 1)
        floor_ms = 1000.0 / fs
        if out["rmssd_ms"] < 2 * floor_ms:
            out["rmssd_note"] = (f"at the {floor_ms:.1f} ms sampling floor - "
                                 "treat as no resolvable variability")
    else:
        out["rmssd_ms"] = None
        out["rmssd_note"] = "not reported: trace too noisy or beats under-detected"

    if not good_coverage:
        out["quality"] = "rate only"
        out["note"] = (f"found {out['n_beats']} beats where the rate implies "
                       f"{out['expected_beats']}; rate is from the spectrum and "
                       "still trustworthy, per-beat timing is not")
    return out


# ---------------------------------------------------------------- plots ------
def make_figures(t, raw, clean, names, f, p, conds, results, fs, figdir, line_hz):
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25})

    # 1. raw vs filtered, first 10 s
    n = min(int(10 * fs), raw.shape[1])
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 7), sharex=True)
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(t[:n], raw[i, :n], lw=0.5, alpha=0.45, label="raw")
        ax.plot(t[:n], clean[i, :n], lw=0.7, label="filtered")
        ax.set_ylabel(f"{names[i]}\n(uV)")
        if i == 0:
            ax.legend(loc="upper right", ncols=2, fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("time (s)")
    fig.suptitle("Raw vs filtered EEG (first 10 s)")
    fig.tight_layout(); fig.savefig(figdir / "01_traces.png"); plt.close(fig)

    # 2. PSD by condition
    uconds = [c for c in ["eyes_open", "eyes_closed", "all"] if c in set(conds)]
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.6), sharey=True)
    for i, ax in enumerate(np.atleast_1d(axes)):
        for c in uconds:
            m = conds == c
            if m.sum() == 0:
                continue
            mean = p[m, i].mean(axis=0)
            sem = p[m, i].std(axis=0) / np.sqrt(m.sum())
            ax.semilogy(f, mean, lw=1.3, label=f"{c} (n={m.sum()})")
            ax.fill_between(f, mean - sem, mean + sem, alpha=0.2)
        ax.axvspan(8, 13, color="orange", alpha=0.12)
        ax.set_xlim(1, 45); ax.set_title(names[i]); ax.set_xlabel("Hz")
        if i == 0:
            ax.set_ylabel("PSD (uV^2/Hz)"); ax.legend(fontsize=8)
    fig.suptitle(f"Power spectral density by condition (alpha band shaded, notch @ {line_hz:.0f} Hz)")
    fig.tight_layout(); fig.savefig(figdir / "02_psd.png"); plt.close(fig)

    # 3. band powers grouped
    if len(uconds) > 1:
        fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.6), sharey=True)
        x = np.arange(len(BANDS)); w = 0.8 / len(uconds)
        for i, ax in enumerate(np.atleast_1d(axes)):
            for j, c in enumerate(uconds):
                vals = [results["band_power_rel"][c][names[i]][b] for b in BANDS]
                ax.bar(x + j * w - 0.4 + w / 2, vals, w, label=c)
            ax.set_xticks(x); ax.set_xticklabels(BANDS.keys(), rotation=45)
            ax.set_title(names[i])
            if i == 0:
                ax.set_ylabel("relative power"); ax.legend(fontsize=8)
        fig.suptitle("Relative band power by condition")
        fig.tight_layout(); fig.savefig(figdir / "03_bandpower.png"); plt.close(fig)

    # 4. alpha time-course spectrogram, posterior mean
    post = [i for i, nm in enumerate(names) if nm in ("TP9", "TP10")] or list(range(len(names)))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ff, tt, Sxx = signal.spectrogram(clean[post].mean(axis=0), fs=fs,
                                     nperseg=int(fs * 2), noverlap=int(fs))
    band = ff <= 45
    ax.pcolormesh(tt, ff[band], 10 * np.log10(Sxx[band] + 1e-12), shading="gouraud", cmap="magma")
    ax.axhline(8, color="w", ls="--", lw=0.8); ax.axhline(13, color="w", ls="--", lw=0.8)
    ax.set_xlabel("time (s)"); ax.set_ylabel("Hz")
    ax.set_title("Posterior (TP9/TP10) spectrogram - alpha band between dashed lines")
    fig.tight_layout(); fig.savefig(figdir / "04_spectrogram.png"); plt.close(fig)


# ----------------------------------------------------------------- main ------
def main():
    ap = argparse.ArgumentParser(description="Analyse a Muse 2 recording.")
    ap.add_argument("run_dir")
    ap.add_argument("--win", type=float, default=2.0, help="epoch length (s)")
    ap.add_argument("--reject-uv", type=float, default=150.0, help="artifact threshold")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    meta, t, raw, marker, names = load(run_dir)
    fs = meta["eeg"]["fs"]
    line_hz = meta.get("line_hz", 50)

    print(f"Run: {run_dir.name}")
    print(f"  {meta['eeg']['n_samples']} samples @ {fs} Hz "
          f"({meta['duration_s']:.0f}s), retained {meta['eeg'].get('retained_pct','?')}%")

    clean = preprocess(raw, fs, line_hz)
    labels = label_samples(t, marker, meta)
    epochs, conds, kept, dropped, peaks = epoch(clean, t, labels, fs,
                                                win_s=args.win, reject_uv=args.reject_uv)
    if kept == 0:
        print(f"\n  No clean EEG epochs: all {dropped} windows exceeded "
              f"{args.reject_uv:.0f} uV.")
        if peaks.size:
            qs = np.percentile(peaks, [10, 50, 90])
            print(f"  Window peak amplitude: median {qs[1]:.0f} uV, "
                  f"10th pct {qs[0]:.0f}, 90th pct {qs[2]:.0f}")
            print(f"  --reject-uv {int(np.ceil(qs[0] / 50) * 50)} would keep "
                  f"about 10% of windows.")
            if qs[1] > 500:
                print("  A median above ~500 uV is an electrode-contact problem, not"
                      " brain activity - raising the threshold analyses noise.")
        print("  Skipping EEG spectral analysis; PPG is unaffected and continues.\n")
    else:
        print(f"  epochs: {kept} kept, {dropped} rejected (>{args.reject_uv:.0f} uV)")
        for c in sorted(set(conds)):
            print(f"    {c}: {(conds == c).sum()}")

    f, p = psd(epochs, fs) if kept else (None, None)

    results = {"run": run_dir.name, "fs": fs, "line_hz": line_hz,
               "epochs_kept": kept, "epochs_rejected": dropped,
               "band_power_abs": {}, "band_power_rel": {}, "contrasts": {}}

    for c in (sorted(set(conds)) if kept else []):
        m = conds == c
        results["band_power_abs"][c] = {}
        results["band_power_rel"][c] = {}
        for i, nm in enumerate(names):
            results["band_power_abs"][c][nm] = {
                b: round(float(np.mean(band_power(f, p[m, i], rng_))), 4)
                for b, rng_ in BANDS.items()}
            results["band_power_rel"][c][nm] = {
                b: round(float(np.mean(band_power(f, p[m, i], rng_, relative=True))), 4)
                for b, rng_ in BANDS.items()}

    # eyes-closed vs eyes-open contrast
    if kept and {"eyes_open", "eyes_closed"} <= set(conds):
        mo, mc = conds == "eyes_open", conds == "eyes_closed"
        print("\n  Alpha (8-13 Hz), eyes-closed vs eyes-open:")
        print(f"    {'chan':6} {'open':>9} {'closed':>9} {'ratio':>7} {'d':>7} {'p':>10}")
        for i, nm in enumerate(names):
            a_o = band_power(f, p[mo, i], BANDS["alpha"])
            a_c = band_power(f, p[mc, i], BANDS["alpha"])
            tstat, pval = stats.ttest_ind(a_c, a_o, equal_var=False)
            d = cohens_d(a_c, a_o)
            ratio = float(np.mean(a_c) / np.mean(a_o)) if np.mean(a_o) > 0 else float("nan")
            results["contrasts"][nm] = {
                "alpha_open": round(float(np.mean(a_o)), 4),
                "alpha_closed": round(float(np.mean(a_c)), 4),
                "ratio_closed_over_open": round(ratio, 3),
                "cohens_d": round(d, 3), "t": round(float(tstat), 3),
                "p_value": float(pval),
            }
            flag = "***" if pval < .001 else "**" if pval < .01 else "*" if pval < .05 else ""
            print(f"    {nm:6} {np.mean(a_o):9.2f} {np.mean(a_c):9.2f} "
                  f"{ratio:7.2f} {d:7.2f} {pval:10.2e} {flag}")
        print("    (uV^2; ratio > 1 = alpha rose with eyes closed, the expected direction)")

    hr = heart_rate(run_dir, meta)
    if hr:
        results["heart_rate"] = hr
        if hr["quality"] == "unusable":
            print(f"\n  PPG: no heart rate - {hr['note']}")
            print(f"       best channel {hr['channel']}, SNR {hr['snr']} "
                  f"(need >= {PPG_SNR_MIN:.0f}). Reseat the band flat on the")
            print("       centre of the forehead and sit still.")
        else:
            print(f"\n  PPG: {hr['bpm_mean']} bpm  "
                  f"[channel {hr['channel']}, SNR {hr['snr']}]")
            print(f"       beats detected {hr['n_beats']} of "
                  f"{hr['expected_beats']} expected "
                  f"(coverage {hr['beat_coverage']})")
            if hr.get("rmssd_ms") is not None:
                print(f"       HRV RMSSD {hr['rmssd_ms']} ms"
                      + (f" - {hr['rmssd_note']}" if hr.get("rmssd_note") else ""))
            else:
                print(f"       HRV {hr.get('rmssd_note','')}")
            if hr["quality"] == "rate only":
                print(f"       NOTE: {hr['note']}")

    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    # tidy long-format CSV for stats software / plotting elsewhere
    rows = ["condition,channel,band,abs_power_uV2,rel_power"]
    for c in results["band_power_abs"]:
        for nm in names:
            for b in BANDS:
                rows.append(f"{c},{nm},{b},{results['band_power_abs'][c][nm][b]},"
                            f"{results['band_power_rel'][c][nm][b]}")
    (run_dir / "band_powers.csv").write_text("\n".join(rows) + "\n")

    if not args.no_figures and kept:
        make_figures(t, raw, clean, names, f, p, conds, results, fs,
                     run_dir / "figures", line_hz)
        print(f"\n  figures -> {run_dir / 'figures'}/")
    print(f"  results -> {run_dir / 'results.json'}, {run_dir / 'band_powers.csv'}")


if __name__ == "__main__":
    main()
