"""Fake headband. Streams plausible EEG to the server over the real protocol.

Exists so the whole system can be built, run and regression-tested with no
hardware in the room. It speaks exactly the wire format the browser does, so
if this works and the browser does not, the fault is in the browser.

    python -m clients.synthetic --duration 60
    python -m clients.synthetic --drop-at 20 --drop-for 5   # test gap handling
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

from pair_eeg.config import EEG
from pair_eeg.transport.protocol import DataFrame

BATCH_S = 0.1


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


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="ws://127.0.0.1:8765")
    p.add_argument("--wearer", default="synthetic")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--drop-at", type=float, default=None, help="seconds until a fake dropout")
    p.add_argument("--drop-for", type=float, default=4.0)
    p.add_argument("--baseline", action="store_true", help="request a baseline block")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    batch_n = int(EEG.rate_hz * BATCH_S)
    counter = 0
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

                # The counter advances through a dropout — that is the point.
                counter += batch_n
                await asyncio.sleep(BATCH_S)
        finally:
            read_task.cancel()

    print(f"sent {sent} frames, {counter} samples over {time.time()-started:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
