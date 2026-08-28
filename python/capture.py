#!/usr/bin/env python3
"""
Muse 2 capture script (BrainFlow, no official app required).

Run this on a LAPTOP with Bluetooth near the headband.

Examples
--------
  python capture.py --check                 # 15s signal-quality check, records nothing
  python capture.py --protocol eyes         # guided eyes-open/closed block protocol
  python capture.py --protocol continuous --duration 300 --label baseline

Output (one folder per run under data/):
  eeg.csv   time_s, TP9, AF7, AF8, TP10, marker
  ppg.csv   time_s, PPG0, PPG1, PPG2
  imu.csv   time_s, ACC_X..Z, GYR_X..Z
  meta.json run parameters, marker legend, dropped-sample estimate
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from brainflow.board_shim import (
    BoardShim,
    BrainFlowInputParams,
    BrainFlowPresets,
    BoardIds,
    BrainFlowError,
)

BOARDS = {
    "muse2": BoardIds.MUSE_2_BOARD,          # native BLE: Windows 10.0.19041+, macOS 10.15+, Linux
    "muse2_bled": BoardIds.MUSE_2_BLED_BOARD,  # BLED112 dongle; deprecated upstream, use muse2
    "muse_s": BoardIds.MUSE_S_BOARD,
    "muse2016": BoardIds.MUSE_2016_BOARD,
}

# Preset codes that turn the ancillary (PPG) stream on. These are firmware
# codes and they differ between models: p61 is a Muse S code, and a Muse 2
# ACCEPTS it silently while streaming no PPG at all - which is why enable_ppg()
# below verifies that samples actually arrive instead of trusting config_board().
PPG_PRESETS = {
    "muse2":      ["p50", "p51"],
    "muse2_bled": ["p50", "p51"],
    "muse_s":     ["p61", "p50"],
    "muse2016":   [],                # no PPG sensor on the 2016 model
}

# Marker codes written into the EEG stream.
MARKERS = {1: "eyes_open_start", 2: "eyes_closed_start", 9: "run_end"}


# ----------------------------------------------------------------- quality ---
def quality_report(eeg_uv, fs, ch_names, line_hz):
    """Per-channel sanity metrics. eeg_uv: (n_ch, n_samples) in microvolts."""
    from scipy.signal import welch

    rows = []
    for name, sig in zip(ch_names, eeg_uv):
        sig = sig[np.isfinite(sig)]
        if sig.size < fs:
            rows.append((name, float("nan"), float("nan"), "NO DATA"))
            continue
        sd = float(np.std(sig))

        # Line-noise ratio: power at 50/60 Hz vs the 1-40 Hz band we care about.
        nper = min(len(sig), int(fs * 2))
        f, p = welch(sig - sig.mean(), fs=fs, nperseg=nper)
        band = (f >= 1) & (f <= 40)
        line = (f >= line_hz - 2) & (f <= line_hz + 2)
        ratio = float(p[line].sum() / p[band].sum()) if p[band].sum() > 0 else float("inf")

        # Thresholds are heuristics tuned for Muse dry electrodes, not hard limits.
        if sd < 1.0:
            verdict = "FLAT - electrode not contacting skin"
        elif sd > 250:
            verdict = "NOISY - loose contact, movement, or hair in the way"
        elif ratio > 0.5:
            verdict = "MAINS - heavy line noise; move from chargers/laptop PSU"
        elif sd > 120:
            verdict = "marginal"
        else:
            verdict = "GOOD"
        rows.append((name, sd, ratio, verdict))

    width = max(len(r[0]) for r in rows)
    print(f"\n  {'chan'.ljust(width)}  {'std (uV)':>9}  {'line/sig':>8}  verdict")
    print(f"  {'-' * width}  {'-' * 9}  {'-' * 8}  {'-' * 40}")
    for name, sd, ratio, verdict in rows:
        print(f"  {name.ljust(width)}  {sd:9.1f}  {ratio:8.2f}  {verdict}")
    return rows


def countdown(seconds, prefix=""):
    for remaining in range(int(seconds), 0, -1):
        print(f"\r  {prefix}{remaining:3d}s ", end="", flush=True)
        time.sleep(1)
    print("\r" + " " * 60 + "\r", end="", flush=True)


# ----------------------------------------------------------------- capture ---
def fetch(board, preset):
    """Pull buffered data for a preset; return empty array if unsupported."""
    try:
        return board.get_board_data(preset=preset)
    except BrainFlowError:
        return np.empty((0, 0))


def enable_ppg(board, codes):
    """Turn on the PPG stream, confirming by data rather than by return code.

    config_board() succeeding proves nothing: a Muse 2 accepts the Muse S
    preset 'p61' without error and then streams no PPG. So each candidate code
    is tried, the stream started, and the ancillary buffer checked for real
    samples. Leaves the stream running on return either way, so EEG still
    records even when no code yields PPG.
    """
    for code in codes:
        try:
            board.config_board(code)
        except BrainFlowError as exc:
            print(f"  preset {code}: rejected ({exc})")
            continue
        board.start_stream(450000)
        time.sleep(3)
        if fetch(board, BrainFlowPresets.ANCILLARY_PRESET).size:
            print(f"  preset {code}: PPG streaming")
            return code, True
        board.stop_stream()
        print(f"  preset {code}: accepted but no PPG data - trying next")
    board.start_stream(450000)
    if codes:
        print("  no preset produced PPG; recording EEG + IMU only")
    return None, False


def save_stream(path, data, ch_rows, ch_names, ts_row, t0, extra_rows=None):
    """Write one preset's buffer to CSV with a relative-time column."""
    if data.size == 0:
        return 0
    ts = data[ts_row]
    cols = {"time_s": ts - t0}
    for row, name in zip(ch_rows, ch_names):
        cols[name] = data[row]
    if extra_rows:
        for row, name in extra_rows:
            cols[name] = data[row]

    header = ",".join(cols.keys())
    arr = np.column_stack(list(cols.values()))
    np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.6f")
    return arr.shape[0]


def main():
    ap = argparse.ArgumentParser(description="Capture Muse 2 data via BrainFlow.")
    ap.add_argument("--board", default="muse2", choices=sorted(BOARDS))
    ap.add_argument("--mac", default="", help="MAC address; speeds up discovery")
    ap.add_argument("--serial-port", default="", help="BLED112 dongle port, e.g. COM3")
    ap.add_argument("--protocol", default="eyes", choices=["eyes", "continuous"])
    ap.add_argument("--duration", type=float, default=300, help="continuous: seconds")
    ap.add_argument("--block", type=float, default=30, help="eyes: seconds per block")
    ap.add_argument("--blocks", type=int, default=4, help="eyes: open/closed pairs")
    ap.add_argument("--check", action="store_true", help="quality check only")
    ap.add_argument("--check-secs", type=float, default=15)
    ap.add_argument("--line-hz", type=float, default=50, help="50 EU / 60 US")
    ap.add_argument("--label", default="run")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--timeout", type=int, default=25, help="BLE discovery seconds")
    ap.add_argument("--ppg-preset", default="", metavar="CODE",
                    help="force a PPG preset code (p50/p51/p61) instead of "
                         "auto-detecting; see PPG_PRESETS")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    BoardShim.enable_dev_board_logger() if args.verbose else BoardShim.disable_board_logger()

    board_id = BOARDS[args.board]
    params = BrainFlowInputParams()
    params.mac_address = args.mac
    params.serial_port = args.serial_port
    params.timeout = args.timeout

    fs_eeg = BoardShim.get_sampling_rate(board_id, BrainFlowPresets.DEFAULT_PRESET)
    eeg_rows = BoardShim.get_eeg_channels(board_id, BrainFlowPresets.DEFAULT_PRESET)
    eeg_names = BoardShim.get_eeg_names(board_id, BrainFlowPresets.DEFAULT_PRESET)
    ts_eeg = BoardShim.get_timestamp_channel(board_id, BrainFlowPresets.DEFAULT_PRESET)
    mk_eeg = BoardShim.get_marker_channel(board_id, BrainFlowPresets.DEFAULT_PRESET)

    board = BoardShim(board_id, params)
    print(f"Connecting to {args.board} (discovery up to {args.timeout}s)...")
    try:
        board.prepare_session()
    except BrainFlowError as exc:
        print(f"\nFAILED to connect: {exc}\n", file=sys.stderr)
        print(
            "Checklist:\n"
            "  - headband ON (LED lit) and not paired/connected to the Muse phone app\n"
            "  - Bluetooth enabled on this machine; try --mac AA:BB:CC:DD:EE:FF\n"
            "  - Windows 10.0.19041+ needs no dongle; --board muse2 uses native BLE\n"
            "  - on Linux you may need: sudo setcap 'cap_net_raw,cap_net_admin+eip' $(which python3)\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Ask the headband for PPG in addition to EEG/IMU.
    codes = [args.ppg_preset] if args.ppg_preset else PPG_PRESETS[args.board]
    print("Enabling ancillary streams...")
    ppg_code, ancillary_ok = enable_ppg(board, codes)

    print("Streaming. Warming up...")
    time.sleep(3)
    # Drain the pre-roll from EVERY preset. get_board_data() with no preset
    # clears only the EEG buffer, which left ppg.csv and imu.csv starting
    # seconds before eeg.csv and the three streams out of alignment.
    for _p in (BrainFlowPresets.DEFAULT_PRESET,
               BrainFlowPresets.ANCILLARY_PRESET,
               BrainFlowPresets.AUXILIARY_PRESET):
        fetch(board, _p)

    # ---- quality check -----------------------------------------------------
    print(f"Signal-quality check ({args.check_secs:.0f}s) - sit still, relax your jaw.")
    countdown(args.check_secs)
    chunk = board.get_board_data(preset=BrainFlowPresets.DEFAULT_PRESET)
    if chunk.size == 0:
        print("\nNo EEG samples arrived. Reseat the headband and retry.", file=sys.stderr)
        board.stop_stream(); board.release_session()
        sys.exit(1)
    quality_report(chunk[eeg_rows], fs_eeg, eeg_names, args.line_hz)

    if args.check:
        print("\nCheck complete (nothing recorded). Re-run without --check to record.")
        board.stop_stream(); board.release_session()
        return

    print("\nIf any channel is FLAT/NOISY, Ctrl-C now, adjust the band, and rerun.")
    print("Tip: dampen the TP9/TP10 pads behind the ears with a little water.")
    countdown(5, "starting in ")

    # ---- recording ---------------------------------------------------------
    board.get_board_data()  # fresh start
    t0 = time.time()
    timeline = []

    try:
        if args.protocol == "eyes":
            print(f"Protocol: {args.blocks} x (open {args.block:.0f}s / closed {args.block:.0f}s)\n")
            for i in range(args.blocks):
                board.insert_marker(1)
                timeline.append({"t_s": time.time() - t0, "marker": 1, "name": MARKERS[1]})
                print(f"Block {i+1}/{args.blocks}  EYES OPEN   - fixate on a point")
                countdown(args.block)

                board.insert_marker(2)
                timeline.append({"t_s": time.time() - t0, "marker": 2, "name": MARKERS[2]})
                print(f"Block {i+1}/{args.blocks}  EYES CLOSED - stay awake, breathe slowly")
                countdown(args.block)
        else:
            print(f"Protocol: continuous, {args.duration:.0f}s\n")
            countdown(args.duration, "recording ")

        board.insert_marker(9)
        timeline.append({"t_s": time.time() - t0, "marker": 9, "name": MARKERS[9]})
        time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupted - saving what was captured so far.")

    elapsed = time.time() - t0

    eeg = fetch(board, BrainFlowPresets.DEFAULT_PRESET)
    ppg = fetch(board, BrainFlowPresets.ANCILLARY_PRESET)
    imu = fetch(board, BrainFlowPresets.AUXILIARY_PRESET)

    board.stop_stream()
    board.release_session()

    # ---- save --------------------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outdir) / f"{stamp}_{args.label}"
    outdir.mkdir(parents=True, exist_ok=True)

    t_ref = eeg[ts_eeg][0] if eeg.size else time.time()

    n_eeg = save_stream(outdir / "eeg.csv", eeg, eeg_rows, eeg_names, ts_eeg, t_ref,
                        extra_rows=[(mk_eeg, "marker")])
    n_ppg = n_imu = 0
    if ppg.size:
        n_ppg = save_stream(
            outdir / "ppg.csv", ppg,
            BoardShim.get_ppg_channels(board_id, BrainFlowPresets.ANCILLARY_PRESET),
            ["PPG0", "PPG1", "PPG2"],
            BoardShim.get_timestamp_channel(board_id, BrainFlowPresets.ANCILLARY_PRESET), t_ref)
    if imu.size:
        acc = BoardShim.get_accel_channels(board_id, BrainFlowPresets.AUXILIARY_PRESET)
        gyr = BoardShim.get_gyro_channels(board_id, BrainFlowPresets.AUXILIARY_PRESET)
        n_imu = save_stream(
            outdir / "imu.csv", imu, acc + gyr,
            ["ACC_X", "ACC_Y", "ACC_Z", "GYR_X", "GYR_Y", "GYR_Z"],
            BoardShim.get_timestamp_channel(board_id, BrainFlowPresets.AUXILIARY_PRESET), t_ref)

    expected = int(elapsed * fs_eeg)
    kept = 100.0 * n_eeg / expected if expected else 0.0

    meta = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "board": args.board,
        "protocol": args.protocol,
        "label": args.label,
        "block_s": args.block if args.protocol == "eyes" else None,
        "blocks": args.blocks if args.protocol == "eyes" else None,
        "duration_s": round(elapsed, 2),
        "line_hz": args.line_hz,
        "ancillary_enabled": ancillary_ok,
        "ppg_preset": ppg_code,
        "eeg": {"fs": fs_eeg, "channels": eeg_names, "n_samples": n_eeg,
                "expected_samples": expected, "retained_pct": round(kept, 1)},
        "ppg": {"fs": BoardShim.get_sampling_rate(board_id, BrainFlowPresets.ANCILLARY_PRESET),
                "n_samples": n_ppg},
        "imu": {"fs": BoardShim.get_sampling_rate(board_id, BrainFlowPresets.AUXILIARY_PRESET),
                "n_samples": n_imu},
        "marker_legend": MARKERS,
        "timeline": timeline,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved to {outdir}/")
    print(f"  eeg.csv  {n_eeg:7d} samples   retained {kept:.1f}% of expected")
    print(f"  ppg.csv  {n_ppg:7d} samples")
    print(f"  imu.csv  {n_imu:7d} samples")
    if n_ppg == 0:
        print("\n  NOTE: no PPG samples, so analyze.py will report no heart rate.")
        print("  Try --ppg-preset p51 (or p50), and make sure the band sits snug")
        print("  and flat on the centre of the forehead - that is where the pulse")
        print("  sensor is. PPG needs firmer contact than EEG does.")
    if kept < 90:
        print("\n  WARNING: heavy packet loss. Move closer to the headband, reduce")
        print("  Bluetooth/Wi-Fi contention, or use a BLED112 dongle.")
    print(f"\nNext:  python analyze.py {outdir}")


if __name__ == "__main__":
    main()
