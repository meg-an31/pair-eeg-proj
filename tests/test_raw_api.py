"""The raw inspection API: see the stream without the analysis in the way."""

import json
from dataclasses import replace

import numpy as np
import pytest
import websockets
from websockets.asyncio.server import serve

from pair_eeg.config import DEFAULT, EEG
from pair_eeg.transport.protocol import DataFrame
from pair_eeg.transport import server as base_server
from pair_eeg.transport.raw_api import make_handler, raw_csv, raw_samples, raw_stats


@pytest.fixture
async def live(tmp_path):
    cfg = replace(DEFAULT, sessions_dir=str(tmp_path), baseline_s=2.0)
    hub = base_server.Hub(cfg)
    async with serve(
        lambda ws: base_server.handle(ws, hub),
        "127.0.0.1", 0, process_request=make_handler(hub),
    ) as srv:
        yield f"127.0.0.1:{srv.sockets[0].getsockname()[1]}", hub


async def _stream(url, hub, blocks=6):
    ws = await websockets.connect(f"ws://{url}")
    await ws.send(json.dumps({"type": "hello", "role": "capture", "wearer": "raw"}))
    await ws.recv()
    rng = np.random.default_rng(7)
    for i in range(blocks):
        samples = rng.normal(0, 15, (256, 4)).astype(np.float32)
        await ws.send(DataFrame(EEG, i * 256, 0.0, samples).encode())
    return ws


def test_idle_when_nothing_streams():
    hub = base_server.Hub(replace(DEFAULT, sessions_dir="/tmp/none"))
    assert raw_samples(hub, "eeg", 256, 1)["type"] == "idle"
    assert raw_stats(hub)["type"] == "idle"
    assert raw_csv(hub, 256) is None


async def test_raw_returns_actual_samples(live):
    url, hub = live
    ws = await _stream(url, hub)
    try:
        import asyncio
        await asyncio.sleep(0.3)
        out = raw_samples(hub, "eeg", 512, 1)
        assert out["type"] == "raw"
        assert out["fs"] == 256.0
        assert out["channels"] == list(EEG.channels)
        assert len(out["samples"]) == 512
        assert len(out["samples"][0]) == 4
        # real microvolts, not zeros
        assert any(v not in (None, 0.0) for v in out["samples"][-1])
    finally:
        await ws.close()


async def test_decimation_reduces_rows(live):
    url, hub = live
    ws = await _stream(url, hub)
    try:
        import asyncio
        await asyncio.sleep(0.3)
        full = raw_samples(hub, "eeg", 512, 1)
        thin = raw_samples(hub, "eeg", 512, 4)
        assert len(thin["samples"]) == len(full["samples"]) // 4
        assert thin["effective_fs"] == 64.0
    finally:
        await ws.close()


async def test_holes_are_null_not_zero(live):
    """A gap and a genuine zero must not look the same."""
    url, hub = live
    ws = await websockets.connect(f"ws://{url}")
    try:
        await ws.send(json.dumps({"type": "hello", "role": "capture", "wearer": "raw"}))
        await ws.recv()
        import asyncio
        await ws.send(DataFrame(EEG, 0, 0.0, np.ones((256, 4), np.float32)).encode())
        # deliberately skip 256..512
        await ws.send(DataFrame(EEG, 512, 0.0, np.ones((256, 4), np.float32)).encode())
        await asyncio.sleep(0.3)

        out = raw_samples(hub, "eeg", 768, 1)
        flat = [v for row in out["samples"] for v in row]
        assert None in flat, "a never-received sample must serialise as null"
        assert out["missing"] > 0
        json.dumps(out)          # must stay valid JSON
    finally:
        await ws.close()


async def test_stats_report_transport_health(live):
    url, hub = live
    ws = await _stream(url, hub)
    try:
        import asyncio
        await asyncio.sleep(0.3)
        stats = raw_stats(hub)
        assert stats["type"] == "raw_stats"
        assert stats["frames_in"] >= 6
        assert stats["expected_fs"] == 256.0
        assert set(stats["rms_uv_1s"]) == set(EEG.channels)
        assert all(v is None or v > 0 for v in stats["rms_uv_1s"].values())
        assert stats["streams"]["eeg"]["available"] > 0
    finally:
        await ws.close()


async def test_csv_has_a_header_and_counters(live):
    url, hub = live
    ws = await _stream(url, hub)
    try:
        import asyncio
        await asyncio.sleep(0.3)
        text = raw_csv(hub, 128)
        lines = text.strip().split("\n")
        assert lines[0] == "sample,TP9,AF7,AF8,TP10"
        assert len(lines) == 129
        assert int(lines[1].split(",")[0]) >= 0
    finally:
        await ws.close()


def test_unknown_stream_is_an_error():
    hub = base_server.Hub(replace(DEFAULT, sessions_dir="/tmp/none2"))
    hub.session = object()      # pretend something is live
    out = raw_samples(hub, "nonsense", 10, 1)
    assert out["type"] == "error"
