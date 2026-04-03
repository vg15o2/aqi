"""
QueueAnalytics — dual-mode zone analytics.

Mode 1 (primary):  Dwell-time tracking — when tracks with IDs are provided,
                   measure per-person time-in-zone directly.
Mode 2 (fallback): Debounce counting + Little's Law — when raw detections
                   only (no tracker), estimate wait via W = L / lambda.

The mode is auto-detected per frame: if dets carry 'track_id', use dwell-time;
otherwise, fall back to debounce + Little's Law.
"""

import json
import time
from collections import deque
from typing import Dict, List, Optional

from config import ALERTS_DIR, log
from zones import in_poly


class QueueAnalytics:
    _ZONE_GRID_X = 5
    _ZONE_GRID_Y = 5

    def __init__(self, alert_thr: int = 10, wait_thr: float = 90.0, cam_id: int = 1):
        self.alert_thr = alert_thr
        self.wait_thr  = wait_thr
        self.cam_id    = cam_id

        # ── Zone polygons (normalised [0..1] coordinates) ────────────────────
        self.zone_queue:   list = []
        self.zone_service: list = []
        self.zone_exit:    list = []

        # ── Tripwire lines (dashboard display only) ──────────────────────────
        self.counter_line: list = []
        self.exit_line:    list = []

        # ── Debounce state (Little's Law fallback) ───────────────────────────
        self.DEBOUNCE   = 3
        self.INIT_PHASE = 5

        self._q_stable: int = 0;  self._q_pend_val: int = None
        self._q_pend_frames: int = 0;  self._q_init_frames: int = 0
        self._s_stable: int = 0;  self._s_pend_val: int = None
        self._s_pend_frames: int = 0;  self._s_init_frames: int = 0

        # ── Rolling zone count samples (~5 min @ 1 fps) ─────────────────────
        self._L_queue_samples:   deque = deque(maxlen=300)
        self._L_service_samples: deque = deque(maxlen=300)

        # ── Flow counts ──────────────────────────────────────────────────────
        self._queue_entries: int = 0
        self._queue_exits:   int = 0
        self._service_exits: int = 0

        # ── Crossing event log ───────────────────────────────────────────────
        self._cross_log: List[Dict] = []

        # ── Session timing ───────────────────────────────────────────────────
        self._t0          = time.time()
        self._frame_count = 0

        # ── Empty-zone counters ──────────────────────────────────────────────
        self._empty_q_frames = 0
        self._empty_s_frames = 0
        self._EMPTY_THRESH   = 10

        # ── Alerts ───────────────────────────────────────────────────────────
        self._ah = deque(maxlen=300)
        self.active_alerts: List[Dict] = []

        # ── Wait / proc time history (Little's Law smoothing) ────────────────
        self._wait_history: deque = deque(maxlen=60)
        self._proc_history: deque = deque(maxlen=60)

        # ── Dwell-time tracking state (Phase C) ─────────────────────────────
        self._track_zone: Dict[int, str] = {}           # track_id -> "queue" | "service"
        self._track_enter_time: Dict[int, float] = {}   # track_id -> entry timestamp
        self._queue_exit_times:   deque = deque(maxlen=300)   # completed dwell durations (seconds)
        self._service_exit_times: deque = deque(maxlen=300)
        self._active_queue_ids:   set = set()
        self._active_service_ids: set = set()
        self._tracking_mode: bool = False

    # ── Configuration ────────────────────────────────────────────────────────
    def set_zones(self, q, s, e):
        self.zone_queue   = q or []
        self.zone_service = s or []
        self.zone_exit    = e or []
        self._q_stable = 0; self._q_pend_val = None; self._q_pend_frames = 0; self._q_init_frames = 0
        self._s_stable = 0; self._s_pend_val = None; self._s_pend_frames = 0; self._s_init_frames = 0
        self._empty_q_frames = 0; self._empty_s_frames = 0

    def set_lines(self, counter_line: list, exit_line: list,
                  entry_dir: int = 1, exit_dir: int = -1):
        self.counter_line = counter_line or []
        self.exit_line    = exit_line or []

    def zones_defined(self) -> bool:
        return bool(self.zone_queue)

    def lines_defined(self) -> bool:
        return bool(self.counter_line)

    def exit_zone_defined(self) -> bool:
        return bool(self.zone_exit) or bool(self.exit_line)

    # ── Multi-anchor zone check ──────────────────────────────────────────────
    def _in_zone(self, bbox, fw, fh, zone_poly) -> bool:
        if not zone_poly or len(zone_poly) < 3:
            return False
        x1n = bbox[0] / fw
        x2n = bbox[2] / fw
        top_y = bbox[1] / fh
        bot_y = bbox[3] / fh
        body_top = top_y + (bot_y - top_y) * 0.33
        bw = x2n - x1n
        margin = bw * 0.10
        lx = x1n + margin
        rx = x2n - margin
        if rx <= lx:
            lx = rx = (x1n + x2n) / 2
        nx = self._ZONE_GRID_X
        ny = self._ZONE_GRID_Y
        for iy in range(ny):
            py = body_top + (bot_y - body_top) * iy / max(ny - 1, 1)
            for ix in range(nx):
                px = lx + (rx - lx) * ix / max(nx - 1, 1)
                if in_poly((px, py), zone_poly):
                    return True
        return False

    # ── Dwell-time helpers ───────────────────────────────────────────────────
    def _recent_rate(self, times_dq: deque, now: float, window: float = 300.0) -> float:
        """Events per minute within a rolling window (default 5 min)."""
        cutoff = now - window
        recent = [t for t in times_dq if t > cutoff]
        if len(recent) < 1:
            return 0.0
        span = now - recent[0]
        if span < 1.0:
            return 0.0
        return len(recent) / (span / 60.0)

    @staticmethod
    def _percentile(vals: list, p: float) -> Optional[float]:
        if not vals:
            return None
        s = sorted(vals)
        k = (len(s) - 1) * (p / 100.0)
        lo = int(k)
        hi = min(lo + 1, len(s) - 1)
        frac = k - lo
        return s[lo] + frac * (s[hi] - s[lo])

    def _dwell_stats(self, times_dq: deque, window: float = 600.0) -> Dict:
        now = time.time()
        cutoff = now - window
        recent = [d for _, d in times_dq if _ > cutoff] if times_dq and isinstance(next(iter(times_dq), None), tuple) else []
        # times_dq stores (exit_timestamp, dwell_seconds) tuples
        # Actually, let's just store dwell seconds and use _queue_exit_times differently
        # Re-check: we store (timestamp, dwell_s) tuples
        vals = []
        for item in times_dq:
            if isinstance(item, (list, tuple)):
                ts, dwell = item
                if ts > cutoff:
                    vals.append(dwell)
            else:
                vals.append(item)
        if not vals:
            return {"avg": None, "p50": None, "p90": None, "min": None, "max": None, "n": 0}
        return {
            "avg": round(sum(vals) / len(vals), 1),
            "p50": round(self._percentile(vals, 50), 1),
            "p90": round(self._percentile(vals, 90), 1),
            "min": round(min(vals), 1),
            "max": round(max(vals), 1),
            "n":   len(vals),
        }

    # ── Main per-frame update ────────────────────────────────────────────────
    def update(self, dets, fw, fh) -> dict:
        now = time.time()
        self._frame_count += 1

        # Auto-detect tracking mode: once we see tracked dets, stay in tracking mode
        # (empty frames don't mean tracking stopped — tracker just has no active tracks)
        if dets and "track_id" in dets[0]:
            self._tracking_mode = True
        has_tracks = self._tracking_mode

        # ── Zone assignment ──────────────────────────────────────────────────
        tz = {}              # bbox_key -> zone_name  (for drawing)
        tz_by_track = {}     # track_id -> zone_name  (for dwell-time)
        L_queue_now = 0
        L_service_now = 0

        for det in dets:
            bbox = det["bbox"]
            bbox_key = (int(bbox[0]), int(bbox[1]))
            tid = det.get("track_id")
            if self._in_zone(bbox, fw, fh, self.zone_service):
                L_service_now += 1
                tz[bbox_key] = "service"
                if tid is not None:
                    tz_by_track[tid] = "service"
            elif self._in_zone(bbox, fw, fh, self.zone_queue):
                L_queue_now += 1
                tz[bbox_key] = "queue"
                if tid is not None:
                    tz_by_track[tid] = "queue"

        # Empty-zone counters
        if L_queue_now == 0:
            self._empty_q_frames += 1
        else:
            self._empty_q_frames = 0
        if L_service_now == 0:
            self._empty_s_frames += 1
        else:
            self._empty_s_frames = 0

        # ── Dwell-time tracking (primary mode when tracker active) ───────────
        predicted_wait_s = None
        predicted_proc_s = None
        pred_wait_method = "none"
        pred_proc_method = "none"
        queue_dwells = {}
        service_dwells = {}

        if has_tracks:
            # Record entry times for new tracks in zones
            current_queue_ids = set()
            current_service_ids = set()

            for tid, zone in tz_by_track.items():
                if zone == "queue":
                    current_queue_ids.add(tid)
                    if tid not in self._track_zone or self._track_zone[tid] != "queue":
                        self._track_enter_time[tid] = now
                        self._track_zone[tid] = "queue"
                elif zone == "service":
                    current_service_ids.add(tid)
                    if tid not in self._track_zone or self._track_zone[tid] != "service":
                        self._track_enter_time[tid] = now
                        self._track_zone[tid] = "service"

            # Detect queue exits (was in queue, no longer)
            departed_q = self._active_queue_ids - current_queue_ids
            for tid in departed_q:
                enter = self._track_enter_time.pop(tid, None)
                if enter is not None:
                    dwell = now - enter
                    if dwell > 0.5:  # ignore sub-500ms flickers
                        self._queue_exit_times.append((now, dwell))
                        self._queue_exits += 1
                        self._cross_log.append({
                            "time": time.strftime("%H:%M:%S"),
                            "type": "queue_exit",
                            "track_id": tid,
                            "dwell_s": round(dwell, 1),
                        })
                self._track_zone.pop(tid, None)

            # Detect service exits
            departed_s = self._active_service_ids - current_service_ids
            for tid in departed_s:
                enter = self._track_enter_time.pop(tid, None)
                if enter is not None:
                    dwell = now - enter
                    if dwell > 0.5:
                        self._service_exit_times.append((now, dwell))
                        self._service_exits += 1
                        self._cross_log.append({
                            "time": time.strftime("%H:%M:%S"),
                            "type": "service_exit",
                            "track_id": tid,
                            "dwell_s": round(dwell, 1),
                        })
                self._track_zone.pop(tid, None)

            # Detect queue entries
            new_q = current_queue_ids - self._active_queue_ids
            self._queue_entries += len(new_q)
            for tid in new_q:
                self._cross_log.append({
                    "time": time.strftime("%H:%M:%S"),
                    "type": "entry",
                    "track_id": tid,
                })

            if len(self._cross_log) > 100:
                self._cross_log = self._cross_log[-100:]

            self._active_queue_ids = current_queue_ids
            self._active_service_ids = current_service_ids

            # ── Predicted wait (dwell-time mode) ─────────────────────────────
            queue_dwells = self._dwell_stats(self._queue_exit_times)
            service_dwells = self._dwell_stats(self._service_exit_times)

            if queue_dwells["avg"] is not None:
                predicted_wait_s = queue_dwells["avg"]
                pred_wait_method = "dwell_time"
            if service_dwells["avg"] is not None:
                predicted_proc_s = service_dwells["avg"]
                pred_proc_method = "dwell_time"

        # ── Debounce + Little's Law (fallback when no tracking) ──────────────
        if not has_tracks:
            self._q_init_frames += 1
            if self._q_init_frames <= self.INIT_PHASE:
                self._q_stable = L_queue_now
                self._q_pend_val = L_queue_now
                self._q_pend_frames = self.DEBOUNCE
            else:
                if L_queue_now == self._q_pend_val:
                    self._q_pend_frames += 1
                else:
                    self._q_pend_val = L_queue_now
                    self._q_pend_frames = 1
                if self._q_pend_frames >= self.DEBOUNCE:
                    delta = self._q_pend_val - self._q_stable
                    if delta < 0:
                        n = abs(delta)
                        self._queue_exits += n
                        for _ in range(n):
                            self._cross_log.append({"time": time.strftime("%H:%M:%S"), "type": "queue_exit"})
                        self._q_stable = self._q_pend_val
                    elif delta > 0:
                        self._queue_entries += delta
                        for _ in range(delta):
                            self._cross_log.append({"time": time.strftime("%H:%M:%S"), "type": "entry"})
                        self._q_stable = self._q_pend_val
                    if len(self._cross_log) > 50:
                        self._cross_log = self._cross_log[-50:]

            self._s_init_frames += 1
            if self._s_init_frames <= self.INIT_PHASE:
                self._s_stable = L_service_now
                self._s_pend_val = L_service_now
                self._s_pend_frames = self.DEBOUNCE
            else:
                if L_service_now == self._s_pend_val:
                    self._s_pend_frames += 1
                else:
                    self._s_pend_val = L_service_now
                    self._s_pend_frames = 1
                if self._s_pend_frames >= self.DEBOUNCE:
                    s_delta = self._s_pend_val - self._s_stable
                    if s_delta < 0:
                        n = abs(s_delta)
                        self._service_exits += n
                        for _ in range(n):
                            self._cross_log.append({"time": time.strftime("%H:%M:%S"), "type": "service_exit"})
                        self._s_stable = self._s_pend_val
                    elif s_delta > 0:
                        self._s_stable = self._s_pend_val
                    if len(self._cross_log) > 50:
                        self._cross_log = self._cross_log[-50:]

        # ── Rolling occupancy ────────────────────────────────────────────────
        self._L_queue_samples.append(L_queue_now)
        self._L_service_samples.append(L_service_now)

        # ── Little's Law (always computed, used as fallback or only source) ──
        elapsed_min = max(0.017, (now - self._t0) / 60.0)
        L_queue = (sum(self._L_queue_samples) / len(self._L_queue_samples)
                   if self._L_queue_samples else float(L_queue_now))
        L_service = (sum(self._L_service_samples) / len(self._L_service_samples)
                     if self._L_service_samples else float(L_service_now))

        lambda_exit    = self._queue_exits   / elapsed_min
        lambda_entry   = self._queue_entries / elapsed_min
        lambda_service = self._service_exits / elapsed_min
        lambda_q = lambda_exit if lambda_exit > 0 else (lambda_entry if lambda_entry > 0 else 0.0)

        _WARMUP_MIN = 0.25
        _MIN_EXITS  = 1
        warmed_up_q = elapsed_min >= _WARMUP_MIN and self._queue_exits >= _MIN_EXITS
        warmed_up_s = elapsed_min >= _WARMUP_MIN and self._service_exits >= _MIN_EXITS

        # ── Final wait time (prefer dwell-time, fallback Little's Law) ───────
        if predicted_wait_s is not None:
            avg_wait_s = predicted_wait_s
            wait_method = pred_wait_method
        elif lambda_q > 0 and warmed_up_q:
            avg_wait_s = (L_queue / lambda_q) * 60.0
            wait_method = "littles_law"
            self._wait_history.append(avg_wait_s)
        elif lambda_q > 0 and not warmed_up_q:
            avg_wait_s = 0.0
            wait_method = "warming_up"
        elif L_queue_now > 0 and self._wait_history:
            avg_wait_s = sum(self._wait_history) / len(self._wait_history)
            wait_method = "estimated"
        else:
            avg_wait_s = 0.0
            wait_method = "no_data" if L_queue_now > 0 else "none"

        if self._empty_q_frames >= self._EMPTY_THRESH:
            avg_wait_s = 0.0
            wait_method = "none"

        # ── Final proc time ──────────────────────────────────────────────────
        if predicted_proc_s is not None:
            avg_proc_s = predicted_proc_s
            proc_method = pred_proc_method
        elif lambda_service > 0 and warmed_up_s:
            avg_proc_s = (L_service / lambda_service) * 60.0
            proc_method = "littles_law"
            self._proc_history.append(avg_proc_s)
        elif lambda_service > 0 and not warmed_up_s:
            avg_proc_s = 0.0
            proc_method = "warming_up"
        elif L_service_now > 0 and self._proc_history:
            avg_proc_s = sum(self._proc_history) / len(self._proc_history)
            proc_method = "estimated"
        else:
            avg_proc_s = 0.0
            proc_method = "no_data" if L_service_now > 0 else "none"

        if self._empty_s_frames >= self._EMPTY_THRESH:
            avg_proc_s = 0.0
            proc_method = "none"

        throughput_hr = lambda_service * 60.0
        open_counter = (avg_wait_s > self.wait_thr and L_queue_now > 0 and lambda_exit == 0)

        # ── Max wait/proc (from dwell-time data) ────────────────────────────
        max_wait = queue_dwells.get("max", 0.0) or 0.0
        max_proc = service_dwells.get("max", 0.0) or 0.0

        # ── Partial waits (people still in queue — how long so far) ──────────
        partial_wait_samples = 0
        if has_tracks:
            for tid in self._active_queue_ids:
                enter = self._track_enter_time.get(tid)
                if enter:
                    partial = now - enter
                    if partial > max_wait:
                        max_wait = round(partial, 1)
                    partial_wait_samples += 1

        # ── Alerts ───────────────────────────────────────────────────────────
        new_a = []

        def _alert(tp, msg, lv):
            last = [a.get("_ts", 0) for a in self._ah
                    if a.get("type") == tp and not a.get("acked")]
            if not last or now - last[-1] > 30:
                idx = len(self._ah)
                a = {
                    "idx": idx, "time": time.strftime("%H:%M:%S"),
                    "message": msg, "level": lv, "type": tp,
                    "_ts": now, "queue_length": L_queue_now,
                    "avg_wait": round(avg_wait_s, 1), "acked": False,
                }
                self._ah.append(a)
                new_a.append(a)
                self._persist_alert(a)

        if L_queue_now > self.alert_thr:
            _alert("queue_length",
                   f"Queue {L_queue_now} > threshold ({self.alert_thr})",
                   "critical" if L_queue_now > self.alert_thr * 1.5 else "warning")
        if avg_wait_s > self.wait_thr and wait_method not in ("none", "no_data"):
            _alert("wait_time",
                   f"Avg wait {avg_wait_s:.0f}s > {self.wait_thr:.0f}s", "warning")
        if open_counter:
            _alert("open_counter",
                   f"No exits detected — consider opening a new counter "
                   f"(Q={L_queue_now}, wait={avg_wait_s:.0f}s)", "critical")
        self.active_alerts = new_a

        return {
            "queue_length":          L_queue_now,
            "L_queue_avg":           round(L_queue, 1),
            "L_service_avg":         round(L_service, 1),
            "avg_waiting_time":      round(avg_wait_s, 1),
            "predicted_wait_s":      round(avg_wait_s, 1),
            "pred_wait_method":      wait_method,
            "open_counter":          open_counter,
            "avg_processing_time":   round(avg_proc_s, 1),
            "predicted_proc_s":      round(avg_proc_s, 1),
            "pred_proc_method":      proc_method,
            "wait_method":           wait_method,
            "proc_method":           proc_method,
            "queue_empty_frames":    self._empty_q_frames,
            "queue_entries":         self._queue_entries,
            "queue_exits":           self._queue_exits,
            "service_exits":         self._service_exits,
            "lambda_entry_pm":       round(lambda_entry, 3),
            "lambda_exit_pm":        round(lambda_exit, 3),
            "lambda_q_pm":           round(lambda_q, 3),
            "lambda_service_pm":     round(lambda_service, 3),
            "counter_crossings":     self._queue_exits,
            "exit_crossings":        self._service_exits,
            "entry_crossings":       self._queue_entries,
            "lambda_counter_pm":     round(lambda_q, 3),
            "throughput_per_hour":   round(throughput_hr, 1),
            "total_processed":       self._service_exits,
            "zone_counts":           {"queue": L_queue_now, "service": L_service_now, "exit": 0},
            "crossing_log":          self._cross_log[-10:],
            "alerts":                new_a,
            "max_waiting_time":      round(max_wait, 1),
            "max_processing_time":   round(max_proc, 1),
            "tput_method":           "observed" if self._service_exits > 0 else "none",
            "seeded_count":          0,
            "partial_wait_samples":  partial_wait_samples,
            "queue_dwells":          queue_dwells,
            "service_dwells":        service_dwells,
            "tracking_mode":         has_tracks,
            "warnings":              [],
            "_tz":                   tz,
        }

    # ── Alert helpers ────────────────────────────────────────────────────────
    def get_alert_history(self) -> List[Dict]:
        return [{k: v for k, v in a.items() if k != "_ts"} for a in self._ah]

    def acknowledge_alert(self, idx: int) -> bool:
        for a in self._ah:
            if a.get("idx") == idx:
                a["acked"] = True
                self._rewrite_alerts_file()
                return True
        return False

    def get_counter_performance(self) -> Dict:
        elapsed_min = max(0.017, (time.time() - self._t0) / 60.0)
        lambda_svc = self._service_exits / elapsed_min
        lambda_q_exit = self._queue_exits / elapsed_min
        L_service = (sum(self._L_service_samples) / len(self._L_service_samples)
                     if self._L_service_samples else 0.0)
        # Prefer dwell-time stats if available
        svc_stats = self._dwell_stats(self._service_exit_times)
        if svc_stats["avg"] is not None:
            avg_proc = svc_stats["avg"]
        elif lambda_svc > 0:
            avg_proc = round((L_service / lambda_svc) * 60.0, 1)
        else:
            avg_proc = None
        return {
            "samples":           self._service_exits,
            "min_s":             svc_stats.get("min"),
            "max_s":             svc_stats.get("max"),
            "avg_s":             avg_proc,
            "p50_s":             svc_stats.get("p50"),
            "p90_s":             svc_stats.get("p90"),
            "total_processed":   self._service_exits,
            "queue_entries":     self._queue_entries,
            "queue_exits":       self._queue_exits,
            "lambda_exit_pm":    round(lambda_q_exit, 3),
            "lambda_service_pm": round(lambda_svc, 3),
            "L_service_avg":     round(L_service, 2),
        }

    def _persist_alert(self, alert: Dict):
        try:
            path = ALERTS_DIR / f"cam_{self.cam_id}_alerts.json"
            existing: List[Dict] = []
            if path.exists():
                try:
                    with open(path) as f:
                        existing = json.load(f)
                except Exception:
                    existing = []
            clean = {k: v for k, v in alert.items() if k != "_ts"}
            existing.append(clean)
            with open(path, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            log.warning(f"[Analytics cam {self.cam_id}] alert persist: {e}")

    def _rewrite_alerts_file(self):
        try:
            path = ALERTS_DIR / f"cam_{self.cam_id}_alerts.json"
            data = [{k: v for k, v in a.items() if k != "_ts"} for a in self._ah]
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log.warning(f"[Analytics cam {self.cam_id}] alerts rewrite: {e}")

    # ── Resets ───────────────────────────────────────────────────────────────
    def reset(self):
        self._q_stable = 0; self._q_pend_val = None; self._q_pend_frames = 0; self._q_init_frames = 0
        self._s_stable = 0; self._s_pend_val = None; self._s_pend_frames = 0; self._s_init_frames = 0
        self._L_queue_samples.clear(); self._L_service_samples.clear()
        self._queue_entries = 0; self._queue_exits = 0; self._service_exits = 0
        self._cross_log.clear()
        self._t0 = time.time(); self._frame_count = 0; self._ah.clear()
        self._wait_history.clear(); self._proc_history.clear()
        self._empty_q_frames = 0; self._empty_s_frames = 0
        # Dwell-time state
        self._track_zone.clear(); self._track_enter_time.clear()
        self._queue_exit_times.clear(); self._service_exit_times.clear()
        self._active_queue_ids.clear(); self._active_service_ids.clear()

    def smooth_reset(self):
        """Loop boundary — preserve cumulative counters, flush pending state."""
        self._q_pend_val = None; self._q_pend_frames = 0
        self._s_pend_val = None; self._s_pend_frames = 0
        self._empty_q_frames = 0; self._empty_s_frames = 0
        # Don't clear dwell-time state — tracks persist across loop boundary
        log.info(
            f"[Analytics cam {self.cam_id}] smooth_reset — "
            f"counters preserved (exits={self._queue_exits} entries={self._queue_entries} "
            f"svc_exits={self._service_exits} active_tracks="
            f"{len(self._active_queue_ids) + len(self._active_service_ids)})"
        )
