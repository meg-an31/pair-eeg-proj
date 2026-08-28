"""Fake headband. Streams plausible EEG to the server over the real protocol.

Exists so the whole system can be built, run and regression-tested with no
hardware in the room. It speaks exactly the wire format the browser does, so
if this works and the browser does not, the fault is in the browser.

    python -m clients.synthetic --duration 60
    python -m clients.synthetic --drop-at 20 --drop-for 5   # test gap handling
    python -m clients.synthetic --no-ppg                    # EEG only

PPG is streamed alongside EEG with a deliberately DIFFERENT counter origin,
because that is what real hardware does: every Muse characteristic carries its
own 16-bit counter and the browser decodes each independently. A client that
started both at zero would hide the alignment bug the server used to have
(`Session._co_window`) instead of exercising it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time

import numpy as np
import websockets

from pair_eeg.config import EEG, PPG
from pair_eeg.transport.protocol import DataFrame

BATCH_S = 0.1


def batch_size(rate_hz: float, frame: int) -> int:
    """Samples in frame `frame`, so the counter advances at the TRUE rate.

    int(rate * BATCH_S) is wrong and not harmlessly so: at 256 Hz and 100 ms
    frames it sends 25 samples where 25.6 are due, so the counter advances at
    250 Hz while the stream claims 256. Alone that is invisible, because
    everything downstream is counter-indexed. Across two streams it is not —
    aligning PPG to EEG means scaling one counter by the ratio of their
    declared rates, and if each stream's counter runs at its own wrong rate
    the two drift apart until the co-window falls off the end of the buffer
    and PPG reads as absent. Differencing the running total keeps the average
    exact and the alignment stable.
    """
    return (int(round(rate_hz * (frame + 1) * BATCH_S))
            - int(round(rate_hz * frame * BATCH_S)))

# Where the fake PPG stream's counter starts. Any value that is not the EEG
# origin will do; a large one also exercises the unwrapped-counter path.
PPG_COUNTER_ORIGIN = 500_000


def generate(n: int, t0: float, fs: float, alpha_gain: float = 1.0) -> np.ndarray:
    """Pink-ish noise plus a 10 Hz alpha bump and 50 Hz mains."""
    t = (np.arange(n) + t0) / fs
    out = np.empty((n, len(EEG.channels)), dtype=np.float32)

    for c in range(len(EEG.channels)):
        pink = np.cumsum(np.random.randn(n)) * 0.6
        pink -= pink.mean()
        alpha = alpha_gain * 12.0 * np.sin(2 * math.pi * 10.0 * t + c * 0.7)
        mains = 3.0 * np.sin(2 * math.pi * 50.0 * t)
        noise = np.random.randn(n) * 2.0
        out[:, c] = pink + alpha + mains + noise

    return out


def beat_schedule(until_s: float, hr_bpm: float, rmssd_ms: float,
                  seed: int = 17) -> np.ndarray:
    """Beat times from t=0 to `until_s`, deterministic for a given seed.

    Regenerated from zero on every call rather than carried as state, so the
    schedule a chunk sees never depends on how the stream was chunked. That
    matters more than it sounds: the first version of this advanced the pulse
    phase within each chunk, which put a beat at every chunk boundary and made
    a broken heart rate of exactly 60 bpm look like a working one.

    Two sources of variation, as in a real heart: respiratory sinus arrhythmia
    (a slow wobble at the breathing rate) and beat-to-beat noise. The noise SD
    is `rmssd/sqrt(2)` because RMSSD is the RMS of SUCCESSIVE DIFFERENCES, and
    differencing two independent draws of SD s gives s*sqrt(2).

    So `rmssd_ms` sets the UNCORRELATED component only. Measured RMSSD comes
    out higher — RSA contributes too, and more at low heart rates, since its
    per-beat slope scales with the square of the beat period. Ground truth for
    a test is this schedule's own RMSSD, not the parameter.
    """
    rng = np.random.default_rng(seed)
    period = 60.0 / hr_bpm
    sd = (rmssd_ms / 1000.0) / math.sqrt(2.0)
    times: list[float] = []
    t = 0.0
    while t <= until_s + 2.0:
        times.append(t)
        rsa = 0.04 * period * math.sin(2 * math.pi * 0.25 * t)
        t += max(period + rsa + float(rng.normal(0.0, sd)), 0.25)
    return np.asarray(times)


def generate_ppg(n: int, t0: float, fs: float, hr_bpm: float,
                 rmssd_ms: float = 40.0) -> np.ndarray:
    """Fake PPG: ambient, IR, red — IR carrying a pulse at `hr_bpm`.

    Each beat is a sharp systolic peak plus a smaller dicrotic notch, because
    the notch is exactly what a naive peak detector counts as a second
    heartbeat; a pure sinusoid would let a broken detector look correct.
    """
    t = (np.arange(n) + t0) / fs
    beats = beat_schedule(float(t[-1]), hr_bpm, rmssd_ms)
    window = beats[(beats > t[0] - 1.5) & (beats < t[-1] + 0.1)]

    pulse = np.zeros(n)
    for bt in window:
        dt = t - bt
        near = (dt >= -0.05) & (dt < 1.2)
        if not near.any():
            continue
        d = dt[near]
        pulse[near] += np.exp(-((d - 0.10) ** 2) / (2 * 0.035**2))
        pulse[near] += 0.35 * np.exp(-((d - 0.32) ** 2) / (2 * 0.05**2))
    pulse *= 1800.0

    out = np.empty((n, len(PPG.channels)), dtype=np.float32)
    out[:, 0] = 40_000.0 + np.random.randn(n) * 30.0           # ambient
    out[:, 1] = 120_000.0 + pulse + np.random.randn(n) * 25.0  # ir
    out[:, 2] = 95_000.0 + 0.7 * pulse                         # red
    return out


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="ws://127.0.0.1:8765")
    p.add_argument("--wearer", default="synthetic")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--drop-at", type=float, default=None, help="seconds until a fake dropout")
    p.add_argument("--drop-for", type=float, default=4.0)
    p.add_argument("--baseline", action="store_true", help="request a baseline block")
    p.add_argument("--hr", type=float, default=68.0, help="fake heart rate, bpm")
    p.add_argument("--no-ppg", action="store_true", help="stream EEG only")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    frame_i = 0
    counter = 0
    ppg_counter = PPG_COUNTER_ORIGIN
    sent = 0
    started = time.time()

    async with websockets.connect(args.url, max_size=2**20) as ws:
        await ws.send(json.dumps({"type": "hello", "role": "capture", "wearer": args.wearer}))

        async def reader() -> None:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                if args.quiet:
                    continue
                if msg.get("type") == "snapshot":
                    print(f"session {msg.get('session')}  stages={msg.get('stages')}")
                elif msg.get("type") == "estimate":
                    q = msg.get("quality", {})
                    axes = msg.get("axes")
                    bits = [
                        f"seq={msg['seq']:>4}",
                        f"{msg['state']:<10}",
                        f"accept={q.get('accept_rate', 0):.2f}",
                    ]
                    if axes:
                        bits.append(" ".join(f"{k[:3]}={v:.2f}" for k, v in axes.items()))
                    if msg.get("baseline"):
                        bits.append(f"baseline {msg['baseline']['progress']:.0%}")
                    print("  ".join(bits))
                elif msg.get("type") == "error":
                    print("server error:", msg.get("detail"))

        read_task = asyncio.create_task(reader())

        if args.baseline:
            await asyncio.sleep(1.0)
            await ws.send(json.dumps({"type": "baseline"}))

        try:
            while time.time() - started < args.duration:
                elapsed = time.time() - started

                dropping = (
                    args.drop_at is not None
                    and args.drop_at <= elapsed < args.drop_at + args.drop_for
                )

                batch_n = batch_size(EEG.rate_hz, frame_i)
                ppg_batch_n = batch_size(PPG.rate_hz, frame_i)
                samples = generate(batch_n, counter, EEG.rate_hz)
                if not dropping:
                    frame = DataFrame(
                        stream=EEG,
                        counter=counter,
                        t_client=time.time() * 1000.0,
                        samples=samples,
                    )
                    await ws.send(frame.encode())
                    sent += 1

                    if not args.no_ppg:
                        await ws.send(
                            DataFrame(
                                stream=PPG,
                                counter=ppg_counter,
                                t_client=time.time() * 1000.0,
                                samples=generate_ppg(
                                    ppg_batch_n,
                                    ppg_counter - PPG_COUNTER_ORIGIN,
                                    PPG.rate_hz,
                                    args.hr,
                                ),
                            ).encode()
                        )
                        sent += 1

                # The counters advance through a dropout — that is the point.
                counter += batch_n
                ppg_counter += ppg_batch_n
                frame_i += 1
                await asyncio.sleep(BATCH_S)
        finally:
            read_task.cancel()

    print(f"sent {sent} frames, {counter} samples over {time.time()-started:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
