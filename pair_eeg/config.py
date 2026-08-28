"""Tunable constants for the streaming pipeline.

Window length and hop are the two numbers that dominate perceived latency:
a WINDOW_S second window is already WINDOW_S/2 behind at its centroid, and
smoothing adds roughly SMOOTHING_WINDOWS * HOP_S on top. Everything else in
the system (network, serialisation, the DSP itself) is noise by comparison.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StreamSpec:
    """One sensor stream as the Muse presents it."""

    name: str
    stream_id: int
    rate_hz: float
    channels: tuple[str, ...]

    @property
    def n_channels(self) -> int:
        return len(self.channels)


EEG = StreamSpec("eeg", 1, 256.0, ("TP9", "AF7", "AF8", "TP10"))
PPG = StreamSpec("ppg", 2, 64.0, ("ambient", "ir", "red"))
IMU = StreamSpec("imu", 3, 52.0, ("ax", "ay", "az", "gx", "gy", "gz"))
THERM = StreamSpec("therm", 4, 16.0, ("temp",))

STREAMS: dict[str, StreamSpec] = {s.name: s for s in (EEG, PPG, IMU, THERM)}
STREAMS_BY_ID: dict[int, StreamSpec] = {s.stream_id: s for s in STREAMS.values()}


@dataclass(frozen=True)
class Config:
    # --- epoching -------------------------------------------------------
    window_s: float = 4.0
    hop_s: float = 1.0

    # --- session lifecycle ----------------------------------------------
    baseline_s: float = 120.0
    fit_check_min_accept: float = 0.6
    fit_check_settle_s: float = 3.0

    # --- quality ---------------------------------------------------------
    accept_rate_window_s: float = 60.0
    degraded_below_accept: float = 0.3
    degraded_after_s: float = 20.0

    # --- slow lane --------------------------------------------------------
    hrv_window_s: float = 60.0
    hrv_refresh_s: float = 5.0

    # --- smoothing ---------------------------------------------------------
    smoothing_windows: float = 5.0

    # --- buffers ------------------------------------------------------------
    buffer_s: float = 180.0

    # --- transport ------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8765

    # --- storage ---------------------------------------------------------------
    sessions_dir: str = "sessions"

    @property
    def smoothing_alpha(self) -> float:
        """EMA coefficient equivalent to averaging over `smoothing_windows`."""
        return 2.0 / (self.smoothing_windows + 1.0)


DEFAULT = Config()
