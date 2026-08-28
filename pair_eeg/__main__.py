"""Run the server.

    python -m pair_eeg [--host H] [--port P] [--sessions DIR]

Both processing stages are null by default — the pipe runs end to end and
reports correctly shaped zeros. See pipeline/processing.py and
pipeline/affect.py for what goes in them.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import replace

from .config import DEFAULT
from .transport.server import run


def main() -> None:
    p = argparse.ArgumentParser(prog="pair_eeg", description=__doc__)
    p.add_argument("--host", default=DEFAULT.host)
    p.add_argument("--port", type=int, default=DEFAULT.port)
    p.add_argument("--sessions", default=DEFAULT.sessions_dir, help="recording directory")
    p.add_argument("--window", type=float, default=DEFAULT.window_s)
    p.add_argument("--hop", type=float, default=DEFAULT.hop_s)
    p.add_argument("--baseline", type=float, default=DEFAULT.baseline_s)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    # A hop or window shorter than one sample makes the epoch loop spin
    # without advancing. Catch it here rather than at 100% CPU.
    min_s = 1.0 / 256.0
    if args.hop < min_s:
        p.error(f"--hop must be at least {min_s:.5f}s (one sample at 256 Hz)")
    if args.window < min_s:
        p.error(f"--window must be at least {min_s:.5f}s")
    if args.window < args.hop:
        p.error("--window must be at least as long as --hop")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = replace(
        DEFAULT,
        host=args.host,
        port=args.port,
        sessions_dir=args.sessions,
        window_s=args.window,
        hop_s=args.hop,
        baseline_s=args.baseline,
    )

    try:
        asyncio.run(run(cfg))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
