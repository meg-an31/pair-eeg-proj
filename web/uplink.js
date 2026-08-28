/* Uplink: batches decoded rows and ships them to the server.
 *
 * Mirrors pair_eeg/transport/protocol.py. The 20-byte header carries the
 * device sample counter of the first sample in the batch — that counter, not
 * wall time, is what the server uses to place samples and spot gaps.
 *
 * When the socket is down, batches spool in memory and are replayed on
 * reconnect with their ORIGINAL counters, so a network dropout produces a
 * marked gap rather than a silent splice.
 */

const MAGIC = 0xee;
const VERSION = 1;
const HEADER_SIZE = 20;

export const STREAM = { EEG: 1, PPG: 2, IMU: 3, THERM: 4 };

export function encodeFrame(streamId, nChannels, counter, samples) {
  const nSamples = samples.length / nChannels;
  const buf = new ArrayBuffer(HEADER_SIZE + samples.length * 4);
  const view = new DataView(buf);

  view.setUint8(0, MAGIC);
  view.setUint8(1, VERSION);
  view.setUint8(2, streamId);
  view.setUint8(3, nChannels);
  view.setUint32(4, counter >>> 0, true);      // little-endian, matches '<'
  view.setUint16(8, nSamples, true);
  view.setUint16(10, 0, true);                 // reserved
  view.setFloat64(12, Date.now(), true);       // drift estimation only

  new Float32Array(buf, HEADER_SIZE).set(samples);
  return buf;
}

export class Uplink extends EventTarget {
  /**
   * @param url           websocket endpoint
   * @param batchMs       how long to accumulate before sending (~100 ms)
   * @param spoolLimit    max batches held while offline (~2 min of EEG)
   */
  constructor(url, { wearer = "unknown", batchMs = 100, spoolLimit = 1500 } = {}) {
    super();
    this.url = url;
    this.wearer = wearer;
    this.batchMs = batchMs;
    this.spoolLimit = spoolLimit;

    this.ws = null;
    this.connected = false;
    this.wantConnected = false;
    this.attempt = 0;
    this.session = null;

    this._spool = [];
    this._pending = [];        // interleaved float samples awaiting a batch
    this._pendingStart = null; // counter of the first pending sample
    this._flushTimer = null;
    this._reconnectTimer = null;

    this.stats = { sent: 0, spooled: 0, dropped: 0, reconnects: 0 };
  }

  connect() {
    this.wantConnected = true;
    this._open();
  }

  _open() {
    this._emit("status", { status: "connecting" });
    let ws;
    try {
      ws = new WebSocket(this.url);
    } catch (err) {
      this._scheduleReconnect();
      return;
    }
    ws.binaryType = "arraybuffer";
    this.ws = ws;

    ws.onopen = () => {
      this.connected = true;
      this.attempt = 0;
      ws.send(JSON.stringify({ type: "hello", role: "capture", wearer: this.wearer }));
      this._replaySpool();
      this._emit("status", { status: "connected" });
    };

    ws.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === "snapshot") this.session = msg.session ?? this.session;
      this._emit("message", msg);
    };

    ws.onclose = () => {
      this.connected = false;
      this.ws = null;
      this._emit("status", { status: "disconnected" });
      if (this.wantConnected) this._scheduleReconnect();
    };

    ws.onerror = () => {
      // onclose always follows; reconnect is handled there.
      this._emit("status", { status: "error" });
    };
  }

  _scheduleReconnect() {
    if (this._reconnectTimer !== null || !this.wantConnected) return;
    const delay = Math.min(500 * 2 ** this.attempt, 10000);
    this.attempt++;
    this._emit("status", { status: "reconnecting", attempt: this.attempt, delayMs: delay });
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this.stats.reconnects++;
      this._open();
    }, delay);
  }

  /** Queue one aligned EEG row. `index` is the absolute sample counter. */
  pushRow(index, samples) {
    if (this._pendingStart === null) {
      this._pendingStart = index;
      this._scheduleFlush();
    } else if (index !== this._pendingStart + this._pending.length / 4) {
      // Non-contiguous: close the current batch so the gap stays visible
      // in the counters rather than being absorbed into one frame.
      this._flush();
      this._pendingStart = index;
      this._scheduleFlush();
    }
    for (let c = 0; c < 4; c++) this._pending.push(samples[c]);
  }

  _scheduleFlush() {
    if (this._flushTimer !== null) return;
    this._flushTimer = setTimeout(() => {
      this._flushTimer = null;
      this._flush();
    }, this.batchMs);
  }

  _flush() {
    if (this._pendingStart === null || this._pending.length === 0) return;
    const frame = encodeFrame(
      STREAM.EEG,
      4,
      this._pendingStart,
      Float32Array.from(this._pending)
    );
    this._pending.length = 0;
    this._pendingStart = null;
    this._send(frame);
  }

  _send(frame) {
    if (this.connected && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(frame);
      this.stats.sent++;
      return;
    }
    this._spool.push(frame);
    this.stats.spooled++;
    if (this._spool.length > this.spoolLimit) {
      this._spool.shift();
      this.stats.dropped++;
    }
  }

  _replaySpool() {
    // Original counters are preserved, so the server places these correctly
    // and sees the dropout as a gap.
    while (this._spool.length && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(this._spool.shift());
      this.stats.sent++;
    }
  }

  send(message) {
    if (this.connected && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
      return true;
    }
    return false;
  }

  marker(label, meta) {
    return this.send({ type: "marker", t: Date.now(), label, meta });
  }

  disconnect() {
    this.wantConnected = false;
    if (this._reconnectTimer !== null) clearTimeout(this._reconnectTimer);
    if (this._flushTimer !== null) clearTimeout(this._flushTimer);
    this._flush();
    this.ws?.close();
  }

  _emit(kind, detail) {
    this.dispatchEvent(new CustomEvent(kind, { detail }));
  }
}
