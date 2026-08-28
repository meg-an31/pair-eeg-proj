"""Raw inspection API — see the stream without the analysis in the way.

When the numbers coming out look wrong, the first question is always whether
the samples going in are any good. This module answers that directly: it
serves the contents of the ring buffer, untouched, so you can look at
microvolts rather than at whatever the pipeline made of them.

It is deliberately a *separate file with its own entry point*. Nothing in the
existing server, session or pipeline is modified — this composes them:

    python -m pair_eeg.transport.raw_api        # same as `python -m pair_eeg`,
                                                # plus the /raw/* routes

Every route the normal server serves still works; these are added in front.

    GET /raw/eeg?n=512&decimate=2   most recent samples, in microvolts
    GET /raw/stats                  counters, gaps, buffer fill, per-channel RMS
    GET /raw/eeg.csv?n=2048         the same samples as a download

Nothing here computes anything. If /raw/eeg shows a plausible waveform and
/latest does not, the fault is downstream of the transport.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import replace
from urllib.parse import parse_qs, urlparse

import numpy as np
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from ..config import DEFAULT, EEG, STREAMS, Config
from . import server as base_server
from .protocol import encode_payload

log = logging.getLogger("pair_eeg.raw_api")

MAX_SAMPLES = 8192
DEFAULT_SAMPLES = 512


# ---------------------------------------------------------------- responses

def _json_response(payload: dict, status: int = 200) -> Response:
    body = encode_payload(payload).encode() + b"\n"
    return Response(
        status,
        "OK" if status == 200 else "Error",
        Headers(
            [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
                ("Access-Control-Allow-Origin", "*"),
                ("Cache-Control", "no-store"),
            ]
        ),
        body,
    )


def _csv_response(text: str, filename: str) -> Response:
    body = text.encode()
    return Response(
        200,
        "OK",
        Headers(
            [
                ("Content-Type", "text/csv"),
                ("Content-Length", str(len(body))),
                ("Content-Disposition", f'attachment; filename="{filename}"'),
                ("Access-Control-Allow-Origin", "*"),
                ("Cache-Control", "no-store"),
            ]
        ),
        body,
    )


def _int_param(query: dict, name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(query.get(name, [default])[0])
    except (ValueError, TypeError):
        return default
    return max(lo, min(hi, value))


# ------------------------------------------------------------------- routes

def raw_samples(hub, stream_name: str, n: int, decimate: int) -> dict:
    """Most recent `n` samples of one stream, straight out of the buffer.

    Values are microvolts for EEG, raw counts for the other streams. Positions
    never received come back as null rather than zero — a hole and a genuine
    zero are different, and papering over that is exactly the confusion this
    endpoint exists to dispel.
    """
    session = hub.session
    if session is None:
        return {"type": "idle", "detail": "nothing is streaming"}

    stream = STREAMS.get(stream_name)
    if stream is None:
        return {"type": "error", "detail": f"unknown stream {stream_name!r}"}

    buf = session.buffers.get(stream.name)
    if buf is None or buf.empty:
        return {
            "type": "raw",
            "session": session.id,
            "stream": stream.name,
            "detail": "no samples received on this stream yet",
            "samples": [],
            "n": 0,
        }

    window = buf.latest(min(n, buf.available() or n))
    samples = window.samples[::decimate] if decimate > 1 else window.samples

    # None, not NaN: NaN is not valid JSON and a hole is not a measurement.
    rows = [
        [None if not np.isfinite(v) else round(float(v), 3) for v in row]
        for row in samples
    ]

    return {
        "type": "raw",
        "session": session.id,
        "state": session.state.value,
        "stream": stream.name,
        "fs": stream.rate_hz,
        "effective_fs": stream.rate_hz / decimate,
        "channels": list(stream.channels),
        "counter": window.start,
        "n": len(rows),
        "decimate": decimate,
        "missing": window.n_missing,
        "fill": round(window.fill_ratio, 4),
        "samples": rows,
    }


def raw_stats(hub) -> dict:
    """Is anything arriving, and is it arriving cleanly?"""
    session = hub.session
    if session is None:
        return {"type": "idle", "capture_held": False, "detail": "nothing is streaming"}

    import time

    last = getattr(session, "_last_ingest", None)
    streams: dict[str, dict] = {}
    for name, buf in session.buffers.items():
        streams[name] = {
            "head": buf.head,
            "tail": buf.tail,
            "available": buf.available(),
            "capacity": buf.capacity,
            "seconds_held": round(buf.available() / STREAMS[name].rate_hz, 2),
            "empty": buf.empty,
        }

    rms: dict[str, float | None] = {}
    eeg = session.buffers[EEG.name]
    if not eeg.empty:
        window = eeg.latest(min(int(EEG.rate_hz), eeg.available()))
        for i, channel in enumerate(EEG.channels):
            column = window.samples[:, i]
            column = column[np.isfinite(column)]
            if column.size:
                centred = column - column.mean()
                rms[channel] = round(float(np.sqrt(np.mean(centred**2))), 2)
            else:
                rms[channel] = None

    return {
        "type": "raw_stats",
        "session": session.id,
        "wearer": session.wearer,
        "state": session.state.value,
        "frames_in": getattr(session, "_frames_in", None),
        "discontinuities": getattr(session, "_discontinuities", None),
        "last_ingest_age_s": round(time.time() - last, 2) if last else None,
        "expected_fs": EEG.rate_hz,
        "streams": streams,
        "rms_uv_1s": rms,
        "viewers": len(hub.viewers),
    }


def raw_csv(hub, n: int) -> str | None:
    session = hub.session
    if session is None:
        return None
    buf = session.buffers[EEG.name]
    if buf.empty:
        return None

    window = buf.latest(min(n, buf.available() or n))
    lines = ["sample," + ",".join(EEG.channels)]
    for offset, row in enumerate(window.samples):
        cells = ["" if not np.isfinite(v) else f"{v:.4f}" for v in row]
        lines.append(f"{window.start + offset}," + ",".join(cells))
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ wiring

def make_handler(hub):
    """Add /raw/* in front of the server's own routes.

    Falls through to the existing handler for everything else, which in turn
    returns None for anything unrecognised so the websocket handshake still
    happens. Composed rather than patched: server.py is untouched.
    """
    fallback = base_server.http_handler(hub)

    def process_request(connection, request):
        parsed = urlparse(request.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)

        if not path.startswith("/raw"):
            return fallback(connection, request)

        if path == "/raw/stats":
            return _json_response(raw_stats(hub))

        if path == "/raw/eeg.csv":
            n = _int_param(query, "n", 2048, 1, MAX_SAMPLES)
            text = raw_csv(hub, n)
            if text is None:
                return _json_response({"type": "idle", "detail": "nothing is streaming"}, 404)
            return _csv_response(text, f"eeg-{n}.csv")

        if path in ("/raw", "/raw/eeg", "/raw/ppg", "/raw/imu", "/raw/therm"):
            stream = "eeg" if path in ("/raw", "/raw/eeg") else path.rsplit("/", 1)[1]
            n = _int_param(query, "n", DEFAULT_SAMPLES, 1, MAX_SAMPLES)
            decimate = _int_param(query, "decimate", 1, 1, 64)
            return _json_response(raw_samples(hub, stream, n, decimate))

        return _json_response({"type": "error", "detail": f"no route {path}"}, 404)

    return process_request


async def run(config: Config, processor=None, affect=None) -> None:
    """Same server, plus the raw routes."""
    hub = base_server.Hub(config, processor=processor, affect=affect)
    async with serve(
        lambda ws: base_server.handle(ws, hub),
        config.host,
        config.port,
        max_size=2**20,
        ping_interval=20,
        ping_timeout=20,
        process_request=make_handler(hub),
    ):
        log.info("listening on ws://%s:%d", config.host, config.port)
        log.info("one capture client, unlimited viewers")
        log.info("polling:    GET /latest  /latest/full  /status  /health")
        log.info("inspection: GET /raw/eeg  /raw/stats  /raw/eeg.csv")
        await asyncio.Future()


def main() -> None:
    p = argparse.ArgumentParser(prog="pair_eeg.raw_api", description=__doc__)
    p.add_argument("--host", default=DEFAULT.host)
    p.add_argument("--port", type=int, default=DEFAULT.port)
    p.add_argument("--sessions", default=DEFAULT.sessions_dir)
    p.add_argument("--window", type=float, default=DEFAULT.window_s)
    p.add_argument("--hop", type=float, default=DEFAULT.hop_s)
    p.add_argument("--baseline", type=float, default=DEFAULT.baseline_s)
    p.add_argument("--null", action="store_true",
                   help="run the placeholder stages instead of the real ones")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    min_s = 1.0 / EEG.rate_hz
    if args.hop < min_s or args.window < min_s or args.window < args.hop:
        p.error("--window must be >= --hop, and both at least one sample long")

    cfg = replace(
        DEFAULT,
        host=args.host,
        port=args.port,
        sessions_dir=args.sessions,
        window_s=args.window,
        hop_s=args.hop,
        baseline_s=args.baseline,
    )
    processor = affect = None
    if not args.null:
        from ..pipeline.affect import MuseAffectMapper
        from ..pipeline.processing import MuseProcessor

        processor = MuseProcessor()
        affect = MuseAffectMapper(processor)

    try:
        asyncio.run(run(cfg, processor=processor, affect=affect))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
