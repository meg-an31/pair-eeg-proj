import numpy as np
import pytest

from pair_eeg.config import EEG, IMU, PPG
from pair_eeg.transport.protocol import (
    HEADER_SIZE,
    DataFrame,
    ProtocolError,
    decode,
    decode_message,
    encode_message,
)


def test_header_is_twenty_bytes():
    assert HEADER_SIZE == 20


@pytest.mark.parametrize("stream", [EEG, PPG, IMU])
def test_round_trip_exact(stream):
    samples = np.random.randn(37, stream.n_channels).astype(np.float32)
    frame = DataFrame(stream=stream, counter=123456, t_client=1.5, samples=samples)
    out = decode(frame.encode())

    assert out.stream is stream
    assert out.counter == 123456
    assert out.t_client == 1.5
    assert out.n_samples == 37
    np.testing.assert_array_equal(out.samples, samples)


def test_end_counter():
    samples = np.zeros((10, EEG.n_channels), dtype=np.float32)
    frame = DataFrame(EEG, 100, 0.0, samples)
    assert frame.end_counter == 110


def test_truncated_frame_rejected():
    samples = np.zeros((4, EEG.n_channels), dtype=np.float32)
    encoded = DataFrame(EEG, 0, 0.0, samples).encode()
    with pytest.raises(ProtocolError):
        decode(encoded[:-4])


def test_short_buffer_rejected():
    with pytest.raises(ProtocolError):
        decode(b"\x00" * 8)


def test_bad_magic_rejected():
    samples = np.zeros((4, EEG.n_channels), dtype=np.float32)
    encoded = bytearray(DataFrame(EEG, 0, 0.0, samples).encode())
    encoded[0] = 0x11
    with pytest.raises(ProtocolError, match="magic"):
        decode(bytes(encoded))


def test_unknown_stream_rejected():
    samples = np.zeros((4, EEG.n_channels), dtype=np.float32)
    encoded = bytearray(DataFrame(EEG, 0, 0.0, samples).encode())
    encoded[2] = 99
    with pytest.raises(ProtocolError, match="stream"):
        decode(bytes(encoded))


def test_zero_samples_round_trips():
    samples = np.zeros((0, EEG.n_channels), dtype=np.float32)
    out = decode(DataFrame(EEG, 5, 0.0, samples).encode())
    assert out.n_samples == 0
    assert out.end_counter == 5


def test_message_round_trip():
    raw = encode_message("hello", role="capture", wearer="meg")
    msg = decode_message(raw)
    assert msg == {"type": "hello", "role": "capture", "wearer": "meg"}


def test_message_requires_type():
    with pytest.raises(ProtocolError):
        decode_message('{"role":"capture"}')


def test_malformed_json_rejected():
    with pytest.raises(ProtocolError):
        decode_message("{not json")
