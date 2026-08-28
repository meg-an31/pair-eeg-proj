"""WebSocket server.

One capture client, many viewers.

The headband allows a single BLE connection, so a second streamer could only
ever be a mistake — a stale tab, a duplicated window, someone else opening the
page. Rather than let two sessions race, the server admits exactly one capture
client and rejects the rest with a message the page can explain.

Viewers are unlimited and read-only. They connect without knowing a session id,
receive a snapshot on arrival, and survive the capture client coming and going,
so a viewer left open overnight picks up tomorrow's session by itself.

    capture  --binary-->  raw sensor frames
             <--json----  estimates
    viewer   <--json----  estimates only
"""

from __future__ import annotations

import asyncio
import logging
import time

import websockets
from websockets.asyncio.server import ServerConnection, serve

from ..config import Config
from ..pipeline.affect import AffectMapper
from ..pipeline.processing import Processor
from ..pipeline.rawlog import RawLog
from ..pipeline.session import Session
from ..pipeline.store import Store, new_session_id
from .protocol import (
    ProtocolError,
    decode,
    decode_message,
    encode_message,
    encode_payload,
)

log = logging.getLogger("pair_eeg.server")

CAPTURE = "capture"
VIEWER = "viewer"


class Hub:
    """Holds the one live session and fans its output out to everyone."""

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

        self.session: Session | None = None
        self.capture_ws: ServerConnection | None = None
        self.capture_since: float | None = None
        self.viewers: set[ServerConnection] = set()

    # ----------------------------------------------------------- capture

    @property
    def capture_held(self) -> bool:
        return self.capture_ws is not None

    async def claim_capture(self, wearer: str, ws: ServerConnection) -> Session | None:
        """Take the capture slot, or return None if it is already held."""
        if self.capture_held:
            return None

        self.capture_ws = ws
        self.capture_since = time.time()

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
            emit=self.broadcast,
            processor=self.processor,
            affect=self.affect,
            raw_log=raw_log,
            session_dir=session_dir,
        )
        self.session = session
        await session.start()

        log.info("capture claimed by %s -> session %s (%s)", wearer, session_id, session_dir)
        await self.tell_viewers(
            "session_started", session=session_id, wearer=wearer
        )
        return session

    async def release_capture(self, ws: ServerConnection) -> None:
        if self.capture_ws is not ws:
            return

        session = self.session

        if session is not None:
            await session.stop()
            if session.raw_log is not None:
                await session.raw_log.close()
            self.store.end_session(session.id)
            log.info("capture released, session %s closed", session.id)

        # Freed only once teardown is complete, so a reconnecting client
        # cannot claim the slot while the previous session is still closing.
        self.capture_ws = None
        self.capture_since = None
        self.session = None

        if session is not None:
            await self.tell_viewers("session_ended", session=session.id)

    # ----------------------------------------------------------- viewers

    def add_viewer(self, ws: ServerConnection) -> None:
        self.viewers.add(ws)
        log.info("viewer joined (%d watching)", len(self.viewers))

    def remove(self, ws: ServerConnection) -> None:
        if ws in self.viewers:
            self.viewers.discard(ws)
            log.info("viewer left (%d watching)", len(self.viewers))

    async def broadcast(self, payload: dict) -> None:
        """Send one payload to the capture client and every viewer."""
        message = encode_payload(payload)
        await self._send_all(message)

    async def tell_viewers(self, kind: str, **fields) -> None:
        await self._send_all(encode_message(kind, **fields), viewers_only=True)

    async def _send_all(self, message: str, viewers_only: bool = False) -> None:
        targets = list(self.viewers)
        if not viewers_only and self.capture_ws is not None:
            targets.append(self.capture_ws)

        if not targets:
            return

        # Concurrently, so one stalled viewer cannot backpressure the epoch
        # loop and slow the stream for everybody else.
        results = await asyncio.gather(
            *(ws.send(message) for ws in targets), return_exceptions=True
        )
        for ws, result in zip(targets, results):
            if isinstance(result, Exception):
                self.viewers.discard(ws)
                if ws is self.capture_ws:
                    log.warning("capture client send failed: %s", result)

    def status(self) -> dict:
        """What a client is told on arrival when there is no live session."""
        return {
            "type": "status",
            "capture_held": self.capture_held,
            "viewers": len(self.viewers),
            "session": self.session.id if self.session else None,
            "uptime_s": round(time.time() - self.capture_since, 1)
            if self.capture_since
            else None,
            "stages": {
                "processing": getattr(self.processor, "name", "null_v0"),
                "affect": getattr(self.affect, "name", "null_v0"),
            },
        }


async def handle(ws: ServerConnection, hub: Hub) -> None:
    role: str | None = None
    session: Session | None = None

    try:
        async for raw in ws:
            # ---- binary: sensor data, capture client only ---------------
            if isinstance(raw, (bytes, bytearray)):
                if role != CAPTURE or session is None:
                    await ws.send(
                        encode_message("error", detail="only the capture client may send data")
                    )
                    continue
                try:
                    session.ingest(decode(bytes(raw)))
                except ProtocolError as exc:
                    await ws.send(encode_message("error", detail=str(exc)))
                continue

            # ---- text: control ------------------------------------------
            try:
                msg = decode_message(raw)
            except ProtocolError as exc:
                await ws.send(encode_message("error", detail=str(exc)))
                continue

            kind = msg.get("type")

            if kind == "hello":
                if role is not None:
                    await ws.send(encode_message("error", detail="already introduced"))
                    continue

                requested = msg.get("role", VIEWER)

                if requested == CAPTURE:
                    session = await hub.claim_capture(msg.get("wearer", "unknown"), ws)
                    if session is None:
                        await ws.send(
                            encode_message(
                                "capture_busy",
                                detail=(
                                    "Another device is already streaming. Only one "
                                    "headband can be connected at a time — close the "
                                    "other capture tab, or join as a viewer instead."
                                ),
                                **{k: v for k, v in hub.status().items() if k != "type"},
                            )
                        )
                        continue
                    role = CAPTURE
                    await ws.send(encode_payload(session.snapshot()))
                else:
                    role = VIEWER
                    hub.add_viewer(ws)
                    if hub.session is not None:
                        await ws.send(encode_payload(hub.session.snapshot()))
                    else:
                        await ws.send(encode_payload(hub.status()))
                continue

            if role is None:
                await ws.send(encode_message("error", detail="send hello first"))
                continue

            if kind == "ping":
                await ws.send(encode_message("pong"))
                continue

            if kind == "status":
                await ws.send(encode_payload(hub.status()))
                continue

            # Everything below changes session state — capture client only.
            if role != CAPTURE or session is None:
                await ws.send(
                    encode_message("error", detail="viewers cannot control the session")
                )
                continue

            if kind == "baseline":
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

            else:
                await ws.send(encode_message("error", detail=f"unknown type {kind!r}"))

    except websockets.ConnectionClosed:
        pass
    finally:
        hub.remove(ws)
        if role == CAPTURE:
            await hub.release_capture(ws)


async def run(
    config: Config,
    processor: Processor | None = None,
    affect: AffectMapper | None = None,
) -> None:
    hub = Hub(config, processor=processor, affect=affect)
    async with serve(
        lambda ws: handle(ws, hub),
        config.host,
        config.port,
        max_size=2**20,
        ping_interval=20,
        ping_timeout=20,
    ):
        log.info("listening on ws://%s:%d", config.host, config.port)
        log.info("one capture client, unlimited viewers")
        log.info(
            "processing=%s  affect=%s",
            getattr(processor, "name", "null_v0"),
            getattr(affect, "name", "null_v0"),
        )
        await asyncio.Future()
