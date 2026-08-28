"""WebSocket server.

One socket per client, used in both directions:

    capture client  --binary-->  raw sensor frames
                    <--json---   estimates (so the capture page can render)
    viewer          <--json---   estimates only

A client declares which it is in a `hello` message. Capture clients own a
session; viewers subscribe to one. Both get the same estimate stream, which
means the capture page and a separate dashboard are the same code path.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import websockets
from websockets.asyncio.server import ServerConnection, serve

from ..config import Config
from ..pipeline.affect import AffectMapper
from ..pipeline.processing import Processor
from ..pipeline.rawlog import RawLog
from ..pipeline.session import Session
from ..pipeline.store import Store, new_session_id
from .protocol import ProtocolError, decode, decode_message, encode_message, encode_payload

log = logging.getLogger("pair_eeg.server")


class Hub:
    """Holds the live sessions and fans estimates out to subscribers."""

    def __init__(
        self,
        config: Config,
        processor: Processor | None = None,
        affect: AffectMapper | None = None,
    ):
        self.cfg = config
        self.processor = processor
        self.affect = affect
        self.store = Store(config.sessions_dir)
        self.sessions: dict[str, Session] = {}
        self.subscribers: dict[str, set[ServerConnection]] = {}

    async def open_session(self, wearer: str, ws: ServerConnection) -> Session:
        session_id = new_session_id()
        session_dir = self.store.create_session(wearer, session_id)

        raw_log = RawLog(session_dir)
        await raw_log.start()
        raw_log.write_meta(
            {
                "session_id": session_id,
                "wearer": wearer,
                "streams": {
                    "eeg": {"fs": 256.0, "channels": ["TP9", "AF7", "AF8", "TP10"]},
                    "ppg": {"fs": 64.0},
                    "imu": {"fs": 52.0},
                },
            }
        )

        session = Session(
            session_id=session_id,
            wearer=wearer,
            config=self.cfg,
            emit=lambda payload: self.broadcast(session_id, payload),
            processor=self.processor,
            affect=self.affect,
            raw_log=raw_log,
            session_dir=session_dir,
        )
        self.sessions[session_id] = session
        self.subscribers.setdefault(session_id, set()).add(ws)
        await session.start()
        log.info("session %s opened for %s -> %s", session_id, wearer, session_dir)
        return session

    async def close_session(self, session: Session) -> None:
        await session.stop()
        if session.raw_log is not None:
            await session.raw_log.close()
        self.store.end_session(session.id)
        self.sessions.pop(session.id, None)
        self.subscribers.pop(session.id, None)
        log.info("session %s closed", session.id)

    async def broadcast(self, session_id: str, payload: dict) -> None:
        peers = self.subscribers.get(session_id)
        if not peers:
            return
        message = encode_payload(payload)
        dead: list[ServerConnection] = []
        for ws in peers:
            try:
                await ws.send(message)
            except websockets.ConnectionClosed:
                dead.append(ws)
        for ws in dead:
            peers.discard(ws)

    def subscribe(self, session_id: str, ws: ServerConnection) -> Session | None:
        session = self.sessions.get(session_id)
        if session is not None:
            self.subscribers.setdefault(session_id, set()).add(ws)
        return session

    def unsubscribe(self, ws: ServerConnection) -> None:
        for peers in self.subscribers.values():
            peers.discard(ws)


async def handle(ws: ServerConnection, hub: Hub) -> None:
    session: Session | None = None
    owns_session = False

    try:
        async for raw in ws:
            # ---- binary: sensor data --------------------------------
            if isinstance(raw, (bytes, bytearray)):
                if session is None:
                    await ws.send(encode_message("error", detail="send hello first"))
                    continue
                try:
                    session.ingest(decode(bytes(raw)))
                except ProtocolError as exc:
                    await ws.send(encode_message("error", detail=str(exc)))
                continue

            # ---- text: control --------------------------------------
            try:
                msg = decode_message(raw)
            except ProtocolError as exc:
                await ws.send(encode_message("error", detail=str(exc)))
                continue

            kind = msg.get("type")

            if kind == "hello":
                role = msg.get("role", "capture")
                if role == "capture":
                    session = await hub.open_session(msg.get("wearer", "unknown"), ws)
                    owns_session = True
                else:
                    session = hub.subscribe(msg.get("session", ""), ws)
                    if session is None:
                        await ws.send(encode_message("error", detail="no such session"))
                        continue
                await ws.send(encode_payload(session.snapshot()))

            elif session is None:
                await ws.send(encode_message("error", detail="send hello first"))

            elif kind == "baseline":
                session.begin_baseline()
                await ws.send(encode_payload(session.snapshot()))

            elif kind == "skip_baseline":
                session.skip_baseline()
                await ws.send(encode_payload(session.snapshot()))

            elif kind == "marker":
                if session.raw_log is not None:
                    session.raw_log.append_event(
                        {"t": msg.get("t"), "label": msg.get("label"), "meta": msg.get("meta")}
                    )

            elif kind == "ping":
                await ws.send(encode_message("pong"))

            else:
                await ws.send(encode_message("error", detail=f"unknown type {kind!r}"))

    except websockets.ConnectionClosed:
        pass
    finally:
        hub.unsubscribe(ws)
        if session is not None and owns_session:
            await hub.close_session(session)


async def run(
    config: Config,
    processor: Processor | None = None,
    affect: AffectMapper | None = None,
) -> None:
    hub = Hub(config, processor=processor, affect=affect)
    async with serve(lambda ws: handle(ws, hub), config.host, config.port, max_size=2**20):
        log.info("listening on ws://%s:%d", config.host, config.port)
        log.info(
            "processing=%s  affect=%s",
            getattr(processor, "name", "null_v0"),
            getattr(affect, "name", "null_v0"),
        )
        await asyncio.Future()
