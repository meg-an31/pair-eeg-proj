"""One capture client, many viewers."""

import asyncio
import json

import pytest
import websockets

from pair_eeg.config import DEFAULT, EEG
from pair_eeg.transport.protocol import DataFrame
from pair_eeg.transport.server import Hub, handle
import numpy as np
from dataclasses import replace
from websockets.asyncio.server import serve


@pytest.fixture
async def server(tmp_path):
    # A short window and hop, pinned here rather than inherited: these are
    # transport tests, and the production geometry is a 30 s epoch stepping
    # 5 s, so a test that feeds a few seconds and waits for an estimate would
    # simply time out on a change that has nothing to do with what it checks.
    cfg = replace(DEFAULT, sessions_dir=str(tmp_path), port=0, baseline_s=2.0,
                  window_s=4.0, hop_s=1.0)
    hub = Hub(cfg)
    async with serve(lambda ws: handle(ws, hub), "127.0.0.1", 0) as srv:
        port = srv.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}", hub


async def hello(ws, role, wearer="test"):
    await ws.send(json.dumps({"type": "hello", "role": role, "wearer": wearer}))
    return json.loads(await ws.recv())


async def test_second_capture_client_rejected(server):
    url, hub = server
    async with websockets.connect(url) as first:
        reply = await hello(first, "capture")
        assert reply["type"] == "snapshot"
        assert hub.capture_held

        async with websockets.connect(url) as second:
            reply2 = await hello(second, "capture")
            assert reply2["type"] == "capture_busy"
            assert "one headband" in reply2["detail"]


async def test_capture_slot_frees_on_disconnect(server):
    url, hub = server
    async with websockets.connect(url) as first:
        await hello(first, "capture")
        assert hub.capture_held
    await asyncio.sleep(0.15)
    assert not hub.capture_held

    async with websockets.connect(url) as second:
        reply = await hello(second, "capture")
        assert reply["type"] == "snapshot"


async def test_many_viewers_allowed(server):
    url, hub = server
    viewers = []
    try:
        for _ in range(5):
            ws = await websockets.connect(url)
            viewers.append(ws)
            reply = await hello(ws, "viewer")
            assert reply["type"] in ("status", "snapshot")
        assert len(hub.viewers) == 5
    finally:
        for ws in viewers:
            await ws.close()


async def test_viewer_cannot_send_data(server):
    url, _ = server
    async with websockets.connect(url) as ws:
        await hello(ws, "viewer")
        frame = DataFrame(EEG, 0, 0.0, np.zeros((4, 4), dtype=np.float32))
        await ws.send(frame.encode())
        reply = json.loads(await ws.recv())
        assert reply["type"] == "error"
        assert "capture client" in reply["detail"]


async def test_viewer_cannot_control_session(server):
    url, _ = server
    async with websockets.connect(url) as ws:
        await hello(ws, "viewer")
        await ws.send(json.dumps({"type": "baseline"}))
        reply = json.loads(await ws.recv())
        assert reply["type"] == "error"
        assert "viewers cannot" in reply["detail"]


async def test_viewers_receive_capture_stream(server):
    url, _ = server
    async with websockets.connect(url) as viewer:
        await hello(viewer, "viewer")
        async with websockets.connect(url) as capture:
            await hello(capture, "capture")
            started = json.loads(await viewer.recv())
            assert started["type"] == "session_started"

            for i in range(12):
                samples = np.random.randn(256, 4).astype(np.float32) * 15
                await capture.send(DataFrame(EEG, i * 256, 0.0, samples).encode())
                await asyncio.sleep(0.05)

            msg = json.loads(await asyncio.wait_for(viewer.recv(), timeout=5))
            assert msg["type"] == "estimate"
            assert "quality" in msg
