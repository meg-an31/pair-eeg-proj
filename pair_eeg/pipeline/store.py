"""Session directories and the metadata index.

Layout, split on the line between what can be recomputed and what cannot:

    sessions/<wearer>/<session_id>/
      raw/{eeg,ppg,imu,therm}.f32   <- precious
      raw/{...}.idx                 <- frame counters, so gaps survive
      meta.json                     <- device, rates, counter origin
      events.jsonl                  <- markers, stimuli, self-report
      baseline.json                 <- frozen stats, written the moment they freeze
      derived/                      <- cache, safe to delete

`derived/` is regenerable by replaying raw through the pipeline. Everything
else is not.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def new_session_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return "s_" + now.strftime("%Y%m%d_%H%M%S")


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.sqlite"
        self._db = sqlite3.connect(self.db_path)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                wearer      TEXT NOT NULL,
                started_at  TEXT NOT NULL,
                ended_at    TEXT,
                state       TEXT,
                path        TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS sessions_wearer ON sessions(wearer, started_at);
            """
        )
        self._db.commit()

    def session_dir(self, wearer: str, session_id: str) -> Path:
        return self.root / wearer / session_id

    def create_session(self, wearer: str, session_id: str) -> Path:
        path = self.session_dir(wearer, session_id)
        path.mkdir(parents=True, exist_ok=True)
        (path / "derived").mkdir(exist_ok=True)
        self._db.execute(
            "INSERT OR REPLACE INTO sessions "
            "(session_id, wearer, started_at, state, path) VALUES (?,?,?,?,?)",
            (
                session_id,
                wearer,
                datetime.now(timezone.utc).isoformat(),
                "connecting",
                str(path),
            ),
        )
        self._db.commit()
        return path

    def update_state(self, session_id: str, state: str) -> None:
        self._db.execute(
            "UPDATE sessions SET state=? WHERE session_id=?", (state, session_id)
        )
        self._db.commit()

    def end_session(self, session_id: str) -> None:
        self._db.execute(
            "UPDATE sessions SET ended_at=? WHERE session_id=?",
            (datetime.now(timezone.utc).isoformat(), session_id),
        )
        self._db.commit()

    def list_sessions(self, wearer: str | None = None, limit: int = 50) -> list[dict]:
        sql = "SELECT session_id, wearer, started_at, ended_at, state, path FROM sessions"
        args: tuple = ()
        if wearer:
            sql += " WHERE wearer=?"
            args = (wearer,)
        sql += " ORDER BY started_at DESC LIMIT ?"
        rows = self._db.execute(sql, (*args, limit)).fetchall()
        cols = ["session_id", "wearer", "started_at", "ended_at", "state", "path"]
        return [dict(zip(cols, r)) for r in rows]

    # -- baseline persistence -------------------------------------------
    # Written the moment the statistics freeze. Losing these to a restart
    # costs the wearer another two minutes of sitting still, which is the
    # most expensive recoverable failure in the system.

    def save_baseline(self, wearer: str, session_id: str, baseline: dict) -> None:
        path = self.session_dir(wearer, session_id) / "baseline.json"
        path.write_text(json.dumps(baseline, indent=2))

    def load_baseline(self, wearer: str, session_id: str) -> dict | None:
        path = self.session_dir(wearer, session_id) / "baseline.json"
        return json.loads(path.read_text()) if path.exists() else None

    def close(self) -> None:
        self._db.close()
