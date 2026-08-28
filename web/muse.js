/* Muse 2 / Muse S connection over Web Bluetooth.
 *
 * The reliability problem here is not the GATT calls, it is everything
 * around them:
 *
 *   - The headband allows ONE exclusive connection. If the Muse phone app
 *     or an OS Bluetooth pairing holds it, we cannot. That failure looks
 *     like a generic timeout, so it is detected and reported specifically.
 *   - requestDevice() needs a user gesture, but getDevices() +
 *     watchAdvertisements() can reconnect to an already-permitted device
 *     without one. That is what makes unattended reconnection possible.
 *   - Notifications stop silently. A dropped link often does not fire
 *     gattserverdisconnected promptly, so a data watchdog is the real
 *     detector.
 *   - Each of the four EEG channels arrives on its own characteristic with
 *     its own 16-bit packet counter, which wraps every ~51 minutes.
 *
 * Emits decoded, channel-aligned rows. Transport is somebody else's job.
 */

const MUSE_SERVICE = 0xfe8d;
const UUID = (short) => `273e${short}-4c4d-454d-96be-f03bac821358`;

const CTRL = UUID("0001");
const BATTERY = UUID("000b");
const EEG_CHARS = ["0003", "0004", "0005", "0006"].map(UUID);
const PPG_CHARS = ["000f", "0010", "0011"].map(UUID);
const IMU_ACC = UUID("000a");
const IMU_GYRO = UUID("0009");

export const CHANNELS = ["TP9", "AF7", "AF8", "TP10"];
export const FS = 256;
const SPP = 12;                 // samples per EEG notification
const LSB = 0.48828125;         // microvolts per least-significant bit
const COUNTER_WRAP = 65536;

// How far behind the newest sample a row may sit before we emit it with
// whatever channels have arrived. ~0.37 s at 256 Hz.
const ALIGN_SLACK = 96;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Build a Muse control command: length-prefixed, newline-terminated. */
function command(text) {
  const bytes = new TextEncoder().encode("X" + text + "\n");
  bytes[0] = bytes.length - 1;
  return bytes;
}

/** Unpack twelve 12-bit big-endian samples from a 18-byte payload. */
function unpack12(bytes) {
  const out = new Array(SPP);
  let o = 0;
  for (let i = 0; i + 2 < bytes.length + 1 && o < SPP; i += 3) {
    out[o++] = (bytes[i] << 4) | (bytes[i + 1] >> 4);
    if (o < SPP) out[o++] = ((bytes[i + 1] & 0x0f) << 8) | bytes[i + 2];
  }
  return out;
}

/** Monotonic sample index from a wrapping 16-bit packet counter. */
class Counter {
  constructor() {
    this.epoch = 0;
    this.last = -1;
  }
  absolute(packetIndex) {
    if (this.last >= 0 && packetIndex < this.last - COUNTER_WRAP / 2) this.epoch++;
    this.last = packetIndex;
    return (this.epoch * COUNTER_WRAP + packetIndex) * SPP;
  }
  reset() {
    this.epoch = 0;
    this.last = -1;
  }
}

export class MuseConnectionError extends Error {
  constructor(message, { kind = "unknown", cause } = {}) {
    super(message);
    this.name = "MuseConnectionError";
    this.kind = kind;
    this.cause = cause;
  }
}

export class MuseClient extends EventTarget {
  /**
   * Events: status, row, ppg, imu, battery, error, gap
   */
  constructor({ enablePpg = true, enableImu = true, watchdogMs = 3000 } = {}) {
    super();
    this.enablePpg = enablePpg;
    this.enableImu = enableImu;
    this.watchdogMs = watchdogMs;

    this.device = null;
    this.server = null;
    this.control = null;

    this.status = "idle";
    this.wantConnected = false;
    this.attempt = 0;

    this._counters = CHANNELS.map(() => new Counter());
    this._rows = new Map();          // absolute sample index -> Float32Array(4)
    this._seen = new Map();          // absolute sample index -> bitmask
    this._newest = -1;
    this._held = new Float32Array(4);
    this._lastData = 0;
    this._watchdog = null;
    this._reconnectTimer = null;
    this._ppgCounters = PPG_CHARS.map(() => new Counter());

    this.stats = { packets: 0, rows: 0, gaps: 0, heldSamples: 0, reconnects: 0 };

    this._onDisconnect = this._onDisconnect.bind(this);
  }

  // ---------------------------------------------------------------- status

  _setStatus(status, detail) {
    this.status = status;
    this.dispatchEvent(new CustomEvent("status", { detail: { status, ...detail } }));
  }

  _fail(message, kind, cause) {
    const err = new MuseConnectionError(message, { kind, cause });
    this.dispatchEvent(new CustomEvent("error", { detail: err }));
    return err;
  }

  // ------------------------------------------------------------ connecting

  static get supported() {
    return typeof navigator !== "undefined" && !!navigator.bluetooth;
  }

  /** Reconnect to an already-permitted headband, no user gesture needed. */
  async reconnectKnown({ timeoutMs = 12000 } = {}) {
    if (!navigator.bluetooth?.getDevices) return null;

    let known = [];
    try {
      known = await navigator.bluetooth.getDevices();
    } catch {
      return null;                       // flag not enabled; caller falls back
    }
    const device = known.find((d) => /muse/i.test(d.name ?? ""));
    if (!device) return null;

    this._setStatus("waiting-for-device", { name: device.name });

    // The device must be advertising before connect() will succeed.
    if (device.watchAdvertisements) {
      const seen = await this._awaitAdvertisement(device, timeoutMs);
      if (!seen) return null;
    }
    this.device = device;
    await this._openGatt();
    return device;
  }

  _awaitAdvertisement(device, timeoutMs) {
    return new Promise((resolve) => {
      const controller = new AbortController();
      const done = (found) => {
        controller.abort();
        clearTimeout(timer);
        device.removeEventListener("advertisementreceived", onAd);
        resolve(found);
      };
      const onAd = () => done(true);
      const timer = setTimeout(() => done(false), timeoutMs);
      device.addEventListener("advertisementreceived", onAd);
      device.watchAdvertisements({ signal: controller.signal }).catch(() => done(false));
    });
  }

  /** Prompt the chooser. MUST be called from a user gesture. */
  async requestDevice() {
    if (!MuseClient.supported) {
      throw this._fail(
        "Web Bluetooth is unavailable. Use Chrome, Edge or Opera on desktop or Android " +
          "over https or localhost. Safari and Firefox do not support it.",
        "unsupported"
      );
    }
    this._setStatus("requesting");
    try {
      this.device = await navigator.bluetooth.requestDevice({
        filters: [{ services: [MUSE_SERVICE] }, { namePrefix: "Muse" }],
        optionalServices: [MUSE_SERVICE],
      });
    } catch (err) {
      if (err?.name === "NotFoundError") {
        throw this._fail("No headband chosen.", "cancelled", err);
      }
      throw this._fail(`Could not open the device chooser: ${err.message}`, "chooser", err);
    }
    await this._openGatt();
    return this.device;
  }

  async connect() {
    this.wantConnected = true;
    const known = await this.reconnectKnown().catch(() => null);
    if (known) return known;
    return this.requestDevice();
  }

  async _openGatt() {
    const device = this.device;
    device.removeEventListener("gattserverdisconnected", this._onDisconnect);
    device.addEventListener("gattserverdisconnected", this._onDisconnect);

    this._setStatus("connecting", { name: device.name });

    try {
      this.server = await device.gatt.connect();
    } catch (err) {
      // The Muse permits one connection. Anything else holding it - the
      // phone app, an OS pairing, another tab - surfaces here as a generic
      // failure, so name the likely cause rather than the symptom.
      throw this._fail(
        "Could not connect. The Muse allows only one connection at a time — " +
          "close the Muse app, unpair it in system Bluetooth settings, and close " +
          "any other tab holding it. Then make sure the headband is on and blinking.",
        "exclusive",
        err
      );
    }

    const service = await this.server.getPrimaryService(MUSE_SERVICE);
    this.control = await service.getCharacteristic(CTRL);

    await this._subscribeControl(service);
    await this._subscribeEeg(service);
    if (this.enablePpg) await this._subscribePpg(service).catch(() => {});
    if (this.enableImu) await this._subscribeImu(service).catch(() => {});

    await this._startStreaming();

    this.attempt = 0;
    this._lastData = performance.now();
    this._startWatchdog();
    this._setStatus("streaming", { name: device.name });
  }

  async _subscribeControl(service) {
    try {
      const battery = await service.getCharacteristic(BATTERY);
      await battery.startNotifications();
      battery.addEventListener("characteristicvaluechanged", (ev) => {
        const pct = ev.target.value.getUint16(2) / 512;
        this.dispatchEvent(new CustomEvent("battery", { detail: { percent: pct } }));
      });
    } catch {
      /* battery is optional */
    }
  }

  async _subscribeEeg(service) {
    for (let ch = 0; ch < EEG_CHARS.length; ch++) {
      const characteristic = await service.getCharacteristic(EEG_CHARS[ch]);
      await characteristic.startNotifications();
      characteristic.addEventListener("characteristicvaluechanged", (ev) =>
        this._onEeg(ch, ev.target.value)
      );
    }
  }

  async _subscribePpg(service) {
    for (let i = 0; i < PPG_CHARS.length; i++) {
      const characteristic = await service.getCharacteristic(PPG_CHARS[i]);
      await characteristic.startNotifications();
      characteristic.addEventListener("characteristicvaluechanged", (ev) =>
        this._onPpg(i, ev.target.value)
      );
    }
  }

  async _subscribeImu(service) {
    const attach = (characteristic, kind) => {
      characteristic.addEventListener("characteristicvaluechanged", (ev) =>
        this._onImu(kind, ev.target.value)
      );
      return characteristic.startNotifications();
    };
    const acc = await service.getCharacteristic(IMU_ACC);
    const gyro = await service.getCharacteristic(IMU_GYRO);
    await attach(acc, "acc");
    await attach(gyro, "gyro");
  }

  async _startStreaming() {
    // p50 = 4x EEG + PPG + IMU. p21 is the EEG-only fallback on units that
    // reject p50. Order matters: halt, select preset, then resume.
    const preset = this.enablePpg ? "p50" : "p21";
    await this.control.writeValue(command("h"));
    await sleep(60);
    try {
      await this.control.writeValue(command(preset));
    } catch {
      await this.control.writeValue(command("p21"));
    }
    await sleep(60);
    await this.control.writeValue(command("s"));
    await sleep(60);
    await this.control.writeValue(command("d"));
  }

  // ---------------------------------------------------------------- decode

  _onEeg(channel, dataView) {
    this.stats.packets++;
    this._lastData = performance.now();

    const packetIndex = dataView.getUint16(0);
    const start = this._counters[channel].absolute(packetIndex);
    const payload = new Uint8Array(dataView.buffer, dataView.byteOffset + 2, dataView.byteLength - 2);
    const raw = unpack12(payload);

    for (let i = 0; i < raw.length; i++) {
      this._pushSample(channel, start + i, LSB * (raw[i] - 0x800));
    }
    this._drain();
  }

  _pushSample(channel, index, microvolts) {
    if (index <= this._newest - ALIGN_SLACK * 4) return;   // hopelessly late
    let row = this._rows.get(index);
    if (row === undefined) {
      row = new Float32Array(4);
      this._rows.set(index, row);
      this._seen.set(index, 0);
    }
    row[channel] = microvolts;
    this._seen.set(index, this._seen.get(index) | (1 << channel));
    if (index > this._newest) this._newest = index;
  }

  /** Emit rows that are complete, or too old to keep waiting for. */
  _drain() {
    const cutoff = this._newest - ALIGN_SLACK;
    const ready = [];

    for (const [index, mask] of this._seen) {
      if (mask === 0b1111 || index <= cutoff) ready.push(index);
    }
    ready.sort((a, b) => a - b);

    for (const index of ready) {
      const row = this._rows.get(index);
      const mask = this._seen.get(index);

      if (mask !== 0b1111) {
        // Sample-and-hold the missing channels rather than zero-filling,
        // which would inject a step edge and read as broadband power.
        for (let c = 0; c < 4; c++) {
          if (!(mask & (1 << c))) {
            row[c] = this._held[c];
            this.stats.heldSamples++;
          }
        }
        this.stats.gaps++;
        this.dispatchEvent(new CustomEvent("gap", { detail: { index, mask } }));
      }
      this._held.set(row);
      this._rows.delete(index);
      this._seen.delete(index);
      this.stats.rows++;
      this.dispatchEvent(new CustomEvent("row", { detail: { index, samples: row } }));
    }
  }

  _onPpg(sensor, dataView) {
    this._lastData = performance.now();
    const packetIndex = dataView.getUint16(0);
    const start = this._ppgCounters[sensor].absolute(packetIndex);
    const values = [];
    for (let i = 2; i + 2 < dataView.byteLength; i += 3) {
      values.push((dataView.getUint8(i) << 16) | (dataView.getUint8(i + 1) << 8) | dataView.getUint8(i + 2));
    }
    this.dispatchEvent(new CustomEvent("ppg", { detail: { sensor, index: start, values } }));
  }

  _onImu(kind, dataView) {
    this._lastData = performance.now();
    const index = dataView.getUint16(0);
    const scale = kind === "acc" ? 0.0000610352 : 0.0074768;
    const samples = [];
    for (let i = 2; i + 5 < dataView.byteLength; i += 6) {
      samples.push([
        dataView.getInt16(i) * scale,
        dataView.getInt16(i + 2) * scale,
        dataView.getInt16(i + 4) * scale,
      ]);
    }
    this.dispatchEvent(new CustomEvent("imu", { detail: { kind, index, samples } }));
  }

  // ----------------------------------------------------------- reliability

  _startWatchdog() {
    this._stopWatchdog();
    this._watchdog = setInterval(() => {
      if (!this.wantConnected) return;
      const quiet = performance.now() - this._lastData;
      if (quiet > this.watchdogMs) {
        // Notifications have stopped without a disconnect event. Treat the
        // link as dead; this is the common failure, not a clean disconnect.
        this._setStatus("stalled", { quietMs: Math.round(quiet) });
        this._stopWatchdog();
        try {
          this.device?.gatt?.disconnect();
        } catch {
          /* already gone */
        }
        this._scheduleReconnect();
      }
    }, Math.max(500, this.watchdogMs / 2));
  }

  _stopWatchdog() {
    if (this._watchdog !== null) {
      clearInterval(this._watchdog);
      this._watchdog = null;
    }
  }

  _onDisconnect() {
    this._stopWatchdog();
    if (!this.wantConnected) {
      this._setStatus("disconnected");
      return;
    }
    this._setStatus("lost");
    this._scheduleReconnect();
  }

  _scheduleReconnect() {
    if (this._reconnectTimer !== null || !this.wantConnected) return;

    // Exponential backoff, capped. Counters are NOT reset: the sample index
    // must stay monotonic across a dropout so the server sees a gap rather
    // than a rewind.
    const delay = Math.min(500 * 2 ** this.attempt, 15000);
    this.attempt++;
    this._setStatus("reconnecting", { attempt: this.attempt, delayMs: delay });

    this._reconnectTimer = setTimeout(async () => {
      this._reconnectTimer = null;
      if (!this.wantConnected) return;
      try {
        if (this.device) {
          await this._openGatt();
          this.stats.reconnects++;
        } else {
          await this.reconnectKnown();
        }
      } catch (err) {
        this._scheduleReconnect();
      }
    }, delay);
  }

  async disconnect() {
    this.wantConnected = false;
    this._stopWatchdog();
    if (this._reconnectTimer !== null) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    try {
      if (this.control) await this.control.writeValue(command("h"));
    } catch {
      /* link may already be gone */
    }
    try {
      this.device?.gatt?.disconnect();
    } catch {
      /* ditto */
    }
    this._setStatus("disconnected");
  }
}
