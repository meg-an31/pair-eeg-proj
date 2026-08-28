#!/usr/bin/env python3
"""
Generate synthetic Muse-2-shaped data in exactly the format capture.py writes.

Purpose: test/develop the analysis pipeline with no headband attached, and give
your group a known-truth dataset to check their numbers against.

  python synth.py --outdir data --label synthetic

Ground truth baked in: alpha (10 Hz) is ~3.5x stronger at TP9/TP10 during the
eyes-closed blocks, i.e. ~10x in POWER. If analyze.py does not recover a
posterior alpha power ratio near 10, the pipeline is wrong.
"""
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FS_EEG, FS_PPG, FS_IMU = 256, 64, 52
CH = ["TP9", "AF7", "AF8", "TP10"]


def pink(n, rng):
    """1/f noise via spectral shaping - the dominant feature of real EEG."""
    spec = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    f = np.arange(len(spec))
    spec /= np.maximum(f, 1) ** 0.5
    spec[0] = 0
    return np.fft.irfft(spec, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--block", type=float, default=30)
    ap.add_argument("--line-hz", type=float, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--label", default="synthetic")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    total_s = args.blocks * 2 * args.block
    n = int(total_s * FS_EEG)
    t = np.arange(n) / FS_EEG

    # eyes-closed mask: block layout is open, closed, open, closed, ...
    closed = (np.floor(t / args.block).astype(int) % 2) == 1

    # alpha amplitude envelope, smoothed so it ramps rather than steps
    env = np.convolve(closed.astype(float), np.hanning(FS_EEG) / np.hanning(FS_EEG).sum(), "same")

    eeg = np.zeros((4, n))
    posterior = {"TP9": 1.0, "TP10": 1.0, "AF7": 0.25, "AF8": 0.25}  # alpha is occipital/posterior

    for i, name in enumerate(CH):
        sig = 22.0 * pink(n, rng) / np.std(pink(n, rng))          # background 1/f
        alpha_amp = 6.0 + 21.0 * env * posterior[name]            # 6 uV base -> ~27 uV closed
        phase = rng.uniform(0, 2 * np.pi)
        sig += alpha_amp * np.sin(2 * np.pi * 10.0 * t + phase)   # 10 Hz alpha
        sig += 3.0 * np.sin(2 * np.pi * args.line_hz * t + rng.uniform(0, 6.28))  # mains
        sig += rng.normal(0, 4.0, n)                              # sensor noise
        eeg[i] = sig

    # eye blinks: large slow deflections, frontal channels only, eyes-open only
    for _ in range(int(total_s / 4)):
        c = rng.integers(0, n)
        if closed[c]:
            continue
        w = int(0.25 * FS_EEG)
        lo, hi = max(0, c - w), min(n, c + w)
        shape = np.hanning(hi - lo) * rng.uniform(80, 160)
        eeg[1, lo:hi] += shape   # AF7
        eeg[2, lo:hi] += shape   # AF8

    # markers at block boundaries
    marker = np.zeros(n)
    timeline = []
    for b in range(args.blocks * 2):
        idx = int(b * args.block * FS_EEG)
        code = 2 if b % 2 else 1
        marker[idx] = code
        timeline.append({"t_s": b * args.block, "marker": code,
                         "name": "eyes_closed_start" if code == 2 else "eyes_open_start"})
    marker[-1] = 9
    timeline.append({"t_s": float(total_s), "marker": 9, "name": "run_end"})

    outdir = Path(args.outdir) / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.label}"
    outdir.mkdir(parents=True, exist_ok=True)

    np.savetxt(outdir / "eeg.csv", np.column_stack([t, *eeg, marker]), delimiter=",",
               header="time_s," + ",".join(CH) + ",marker", comments="", fmt="%.6f")

    # PPG: ~62 bpm pulse train
    tp = np.arange(int(total_s * FS_PPG)) / FS_PPG
    hr = 62 / 60
    pulse = 1800 + 140 * np.sin(2 * np.pi * hr * tp) + 45 * np.sin(4 * np.pi * hr * tp)
    ppg = np.column_stack([tp] + [pulse + rng.normal(0, 12, len(tp)) + o for o in (0, 300, 600)])
    np.savetxt(outdir / "ppg.csv", ppg, delimiter=",",
               header="time_s,PPG0,PPG1,PPG2", comments="", fmt="%.6f")

    # IMU: near-still head with slow drift
    ti = np.arange(int(total_s * FS_IMU)) / FS_IMU
    imu = np.column_stack([
        ti,
        rng.normal(0, .01, len(ti)), rng.normal(0, .01, len(ti)), 1 + rng.normal(0, .01, len(ti)),
        rng.normal(0, .3, len(ti)), rng.normal(0, .3, len(ti)), rng.normal(0, .3, len(ti)),
    ])
    np.savetxt(outdir / "imu.csv", imu, delimiter=",",
               header="time_s,ACC_X,ACC_Y,ACC_Z,GYR_X,GYR_Y,GYR_Z", comments="", fmt="%.6f")

    (outdir / "meta.json").write_text(json.dumps({
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "board": "SYNTHETIC", "protocol": "eyes", "label": args.label,
        "block_s": args.block, "blocks": args.blocks,
        "duration_s": total_s, "line_hz": args.line_hz, "ancillary_enabled": True,
        "eeg": {"fs": FS_EEG, "channels": CH, "n_samples": n,
                "expected_samples": n, "retained_pct": 100.0},
        "ppg": {"fs": FS_PPG, "n_samples": len(tp)},
        "imu": {"fs": FS_IMU, "n_samples": len(ti)},
        "marker_legend": {"1": "eyes_open_start", "2": "eyes_closed_start", "9": "run_end"},
        "timeline": timeline,
        "synthetic_ground_truth": {
            "alpha_hz": 10.0,
            "posterior_amplitude_ratio_closed_over_open": "~3.5x at TP9/TP10",
            "expected_alpha_POWER_ratio": "~10x (power scales as amplitude squared)",
            "note": "NOT real EEG - for pipeline validation only",
        },
    }, indent=2))

    print(f"Synthetic dataset written to {outdir}/")
    print(f"  {total_s:.0f}s, {args.blocks} open/closed pairs, {n} EEG samples")
    print(f"\nNext:  python analyze.py {outdir}")


if __name__ == "__main__":
    main()
