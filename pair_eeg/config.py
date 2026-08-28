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
    # muse's native geometry: it scores a 30 s epoch, stepping 5 s, and cuts
    # that epoch into 29 overlapping 2 s windows internally. Two of its gates
    # depend on the length — the tonic/phasic muscle split needs at least 5
    # windows, and the heart features at least 20 beats — so a shorter epoch
    # does not degrade the score gracefully, it silently drops voters.
    #
    # Note what the epoch-level quality gate does and does not do at this
    # length: a 30 s window almost always contains a blink, so the frontal
    # channels get flagged `noisy` while TP9/TP10 keep the epoch accepted.
    # That is correct here — rejecting artifacts at 2 s granularity is muse's
    # job, and it does it per window.
    window_s: float = 30.0
    hop_s: float = 5.0

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
    # 1.0 = pass-through (alpha of 1.0). muse does not smooth: a 30 s epoch is
    # already a heavy average, and an EMA over 5 of them would add ~25 s of
    # lag on top of the 15 s the window centroid already costs.
    smoothing_windows: float = 1.0

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
