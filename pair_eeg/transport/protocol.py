"""Wire format.

Two kinds of message share one WebSocket:

  * binary  -> sensor data, upstream only. Fixed 20-byte header, then
               float32 samples interleaved by channel.
  * text    -> JSON control and results, both directions.

The header carries the device sample counter of the frame's first sample.
That counter, not wall-clock time, is the authoritative clock: window
boundaries and gap detection are computed from it. `t_client` rides along
purely to estimate drift and to align event markers against samples.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import STREAMS_BY_ID, StreamSpec

MAGIC = 0xEE
VERSION = 1

# magic, version, stream_id, n_channels, counter, n_samples, reserved, t_client
_HEADER = struct.Struct("<BBBBIHHd")
HEADER_SIZE = _HEADER.size  # 20


class ProtocolError(ValueError):
    """Raised when a frame cannot be parsed."""


@dataclass(frozen=True)
class DataFrame:
    """A batch of samples for one stream.

    `samples` has shape (n_samples, n_channels) and dtype float32.
    """

    stream: StreamSpec
    counter: int
    t_client: float
    samples: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def end_counter(self) -> int:
        """Counter of the first sample *after* this frame."""
        return self.counter + self.n_samples

    def encode(self) -> bytes:
        arr = np.ascontiguousarray(self.samples, dtype=np.float32)
        head = _HEADER.pack(
            MAGIC,
            VERSION,
            self.stream.stream_id,
            self.stream.n_channels,
            self.counter & 0xFFFFFFFF,
            arr.shape[0],
            0,
            self.t_client,
        )
        return head + arr.tobytes()


def decode(buf: bytes) -> DataFrame:
    """Parse one binary data frame. Raises ProtocolError on anything odd."""
    if len(buf) < HEADER_SIZE:
        raise ProtocolError(f"frame too short: {len(buf)} bytes")

    magic, version, stream_id, n_chan, counter, n_samp, _, t_client = _HEADER.unpack_from(buf)

    if magic != MAGIC:
        raise ProtocolError(f"bad magic 0x{magic:02x}")
    if version != VERSION:
        raise ProtocolError(f"unsupported version {version}")

    stream = STREAMS_BY_ID.get(stream_id)
    if stream is None:
        raise ProtocolError(f"unknown stream id {stream_id}")
    if n_chan != stream.n_channels:
        raise ProtocolError(
            f"{stream.name}: expected {stream.n_channels} channels, frame says {n_chan}"
        )

    expected = HEADER_SIZE + n_samp * n_chan * 4
    if len(buf) != expected:
        raise ProtocolError(f"{stream.name}: expected {expected} bytes, got {len(buf)}")

    samples = np.frombuffer(buf, dtype=np.float32, offset=HEADER_SIZE).reshape(n_samp, n_chan)
    return DataFrame(stream=stream, counter=counter, t_client=t_client, samples=samples)


# --------------------------------------------------------------------------
# Text messages
# --------------------------------------------------------------------------

def encode_message(kind: str, **fields: Any) -> str:
    return json.dumps({"type": kind, **fields}, separators=(",", ":"))


def encode_payload(payload: dict[str, Any]) -> str:
    """Serialise a payload that already carries its own "type".

    Does not mutate the input — payloads are handed to several subscribers.
    """
    if "type" not in payload:
        raise ProtocolError("payload has no 'type'")
    return json.dumps(payload, separators=(",", ":"))


def decode_message(raw: str) -> dict[str, Any]:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"malformed JSON: {exc}") from exc
    if not isinstance(msg, dict) or "type" not in msg:
        raise ProtocolError("message must be an object with a 'type'")
    return msg
