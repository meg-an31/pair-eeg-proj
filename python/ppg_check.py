#!/usr/bin/env python3
"""
Look at a PPG recording and say whether a heartbeat is actually in there.

  python ppg_check.py data/20260828-153929_hr_test

Prints a per-channel quality table, then writes ppg_check.png so you can SEE
the pulse waveform with the detected beats marked. Use it when analyze.py
reports no heart rate, or a rate you do not believe: the figure tells you
immediately whether the problem is sensor contact (no periodic wave at all) or
the beat detector (clear wave, marks in the wrong places).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from analyze import (PPG_BAND, PPG_SNR_MIN, PPG_SNR_HRV,
                     ppg_pulsatile, ppg_channels, heart_rate)


def main():
    ap = argparse.ArgumentParser(description="Diagnose a PPG recording.")
    ap.add_argument("run_dir")
    ap.add_argument("--window", type=float, default=15.0,
                    help="seconds of waveform to plot (default 15)")
    ap.add_argument("--start", type=float, default=30.0,
                    help="offset into the recording to plot from (default 30)")
    args = ap.parse_args()

    run = Path(args.run_dir)
    meta = json.loads((run / "meta.json").read_text())
    path = run / "ppg.csv"
    if not path.exists():
        raise SystemExit(f"No ppg.csv in {run}. PPG did not stream - re-record "
                         "with capture.py, and check the preset line it prints.")

    d = np.genfromtxt(path, delimiter=",", names=True)
    fs = meta.get("ppg", {}).get("fs", 64)
    chans = ppg_channels(d)
    secs = len(d) / fs

    print(f"Run: {run.name}")
    print(f"  ppg.csv: {len(d)} samples @ {fs} Hz = {secs:.1f} s, "
          f"channels {', '.join(chans)}\n")

    print(f"  {'chan':6} {'range':>12} {'sd':>9} {'bpm':>7} {'SNR':>9}  verdict")
    best = None
    for c in chans:
        x = np.asarray(d[c], dtype=float)
        rng = float(np.nanmax(x) - np.nanmin(x))
        r = ppg_pulsatile(x, fs)
        if r is None:
            print(f"  {c:6} {rng:12.0f} {np.nanstd(x):9.1f} {'-':>7} {'-':>9}  "
                  "too short to judge")
            continue
        hz, snr = r
        verdict = ("clean" if snr >= PPG_SNR_HRV else
                   "usable" if snr >= PPG_SNR_MIN else "no pulse found")
        print(f"  {c:6} {rng:12.0f} {np.nanstd(x):9.1f} {60*hz:7.1f} {snr:9.1f}  {verdict}")
        if best is None or snr > best[0]:
            best = (snr, hz, c)

    if best is None:
        raise SystemExit("\n  Not enough PPG data to analyse.")

    snr, hz, chan = best
    bpm = 60 * hz
    print(f"\n  Best channel: {chan} at {bpm:.1f} bpm (SNR {snr:.1f}, "
          f"threshold {PPG_SNR_MIN:.0f})")

    hr = heart_rate(run, meta)
    if hr and hr.get("bpm_mean"):
        print(f"  analyze.py would report: {hr['bpm_mean']} bpm, "
              f"quality '{hr['quality']}', "
              f"{hr.get('n_beats')} of {hr.get('expected_beats')} beats")
    elif hr:
        print(f"  analyze.py would report no rate: {hr.get('note','')}")

    if snr < PPG_SNR_MIN:
        print("\n  No cardiac rhythm stands out on any channel. That is a sensor")
        print("  contact problem, not an analysis problem: the pulse sensor sits in")
        print("  the CENTRE of the forehead band and needs firm, flat skin contact.")
        print("  Push hair aside, seat the band snugly, sit still, and re-record.")

    # ---- figure ------------------------------------------------------------
    x = np.asarray(d[chan], dtype=float)
    t = np.asarray(d[d.dtype.names[0]], dtype=float)
    sos = signal.butter(3, PPG_BAND, btype="bandpass", fs=fs, output="sos")
    xf = signal.sosfiltfilt(sos, signal.detrend(x))
    mad = float(np.median(np.abs(xf - np.median(xf))))
    peaks, _ = signal.find_peaks(xf, distance=int(0.5 * (60.0 / bpm) * fs),
                                 prominence=max(1.4826 * mad * 0.6, 1e-9))

    i0 = int(np.clip(args.start, 0, max(0, secs - args.window)) * fs)
    i1 = int(min(len(x), i0 + args.window * fs))
    sel = (peaks >= i0) & (peaks < i1)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8))
    fig.suptitle(f"{run.name} - PPG diagnostic ({chan}, {bpm:.1f} bpm, SNR {snr:.0f})")

    axes[0].plot(t[i0:i1], x[i0:i1], lw=0.8, color="#444")
    axes[0].set_title(f"raw {chan}, {args.window:.0f} s window - drift and steps "
                      "are normal, look for periodicity")
    axes[0].set_ylabel("raw units")

    axes[1].plot(t[i0:i1], xf[i0:i1], lw=0.9, color="#1f77b4")
    axes[1].plot(t[peaks[sel]], xf[peaks[sel]], "rv", ms=6, label="detected beat")
    axes[1].set_title("band-passed 0.7-3.5 Hz with detected beats - you should see "
                      "one clear bump per beat, each marked once")
    axes[1].set_ylabel("filtered")
    axes[1].legend(loc="upper right", fontsize=8)

    f, pxx = signal.welch(signal.detrend(x[np.isfinite(x)]), fs=fs,
                          nperseg=int(min(len(x), 32 * fs)))
    m = f <= 6
    axes[2].semilogy(f[m], pxx[m], lw=1, color="#2ca02c")
    axes[2].axvspan(*PPG_BAND, alpha=0.12, color="tab:green",
                    label="cardiac band 0.7-3.5 Hz")
    axes[2].axvline(hz, color="tab:red", ls="--", lw=1, label=f"{bpm:.1f} bpm")
    axes[2].set_title("spectrum - a real pulse is a sharp spike inside the shaded band")
    axes[2].set_xlabel("Hz"); axes[2].set_ylabel("power")
    axes[2].legend(loc="upper right", fontsize=8)

    for a in axes:
        a.grid(alpha=0.25)
    fig.tight_layout()
    out = run / "ppg_check.png"
    fig.savefig(out, dpi=110)
    print(f"\n  figure -> {out}")


if __name__ == "__main__":
    main()
