"""
CameraStream and CameraManager — per-camera lifecycle, tracking, drawing, metrics.
"""

import csv
import io
import json
import threading
import time
from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import (
    ALERTS_DIR, DEFAULT_DET_FILTER,
    TRACKER_TRACK_THRESH, TRACKER_MATCH_THRESH,
    TRACKER_TRACK_BUFFER, TRACKER_LOW_THRESH,
    TRACKER_BBOX_INFLATE,
    log,
)
from inference import InferenceEngine
from analytics import QueueAnalytics
from decoder import VideoDecoder
from bytetrack import ByteTracker


# ── Colors ───────────────────────────────────────────────────────────────────
ZONE_COLORS = {
    "queue": (0, 200, 255),      "service": (0, 255, 128),
    "exit": (200, 80, 255),
    "queue_2": (0, 165, 255),    "service_2": (80, 255, 80),
}
PERSON_C = {
    "queue": (0, 200, 255),      "service": (0, 255, 128),
    "queue_2": (0, 165, 255),    "service_2": (80, 255, 80),
    "exit": (200, 80, 255),      None: (180, 180, 180),
}

# Stable color palette for track IDs
_TRACK_PALETTE = [
    (230, 100, 50), (50, 180, 230), (100, 230, 50), (230, 50, 180),
    (50, 230, 180), (180, 50, 230), (230, 180, 50), (50, 100, 230),
    (180, 230, 50), (230, 50, 100), (50, 230, 100), (100, 50, 230),
]

_EMPTY_METRICS = {
    "queue_length": 0, "avg_waiting_time": 0, "avg_processing_time": 0,
    "throughput_per_hour": 0, "tput_method": "none", "total_processed": 0,
    "counter_crossings": 0, "exit_crossings": 0, "entry_crossings": 0,
    "lambda_counter_pm": 0, "lambda_exit_pm": 0,
    "L_queue_avg": 0, "L_service_avg": 0,
    "wait_method": "none", "proc_method": "none",
    "open_counter": False,
    "alerts": [], "zone_counts": {}, "crossing_log": [],
    "warnings": ["Draw zones in Zone Setup tab"],
    "queue_dwells": {}, "service_dwells": {}, "seeded_count": 0,
    "partial_wait_samples": 0, "max_waiting_time": 0, "max_processing_time": 0,
    "tracking_mode": False, "_tz": {},
}


class CameraStream:
    _id_counter = 1

    def __init__(self, name: str, source: str, mode: str = "live",
                 alert_thr: int = 10, wait_thr: float = 90.0, inf_fps: float = 15.0):
        self.cam_id = CameraStream._id_counter
        CameraStream._id_counter += 1
        self.name = name
        self.source = source
        self.mode = mode
        self.inf_fps_limit: float = max(1.0, min(30.0, float(inf_fps)))

        self.decoder = VideoDecoder()
        self.analytics = QueueAnalytics(alert_thr, wait_thr, cam_id=self.cam_id)
        self.analytics_2 = QueueAnalytics(alert_thr, wait_thr, cam_id=self.cam_id)

        # Per-camera tracker with bbox doubling
        self.tracker = ByteTracker(
            track_thresh=TRACKER_TRACK_THRESH,
            match_thresh=TRACKER_MATCH_THRESH,
            track_buffer=TRACKER_TRACK_BUFFER,
            low_thresh=TRACKER_LOW_THRESH,
            bbox_inflate=TRACKER_BBOX_INFLATE,
        )
        self.tracking_enabled: bool = True

        # Per-camera detection filters
        self.det_filter: dict = dict(DEFAULT_DET_FILTER)

        self._lock = threading.Lock()
        self._metrics: Dict = {}
        self._ann: Optional[np.ndarray] = None
        self._inf_fps: float = 0.0
        self._cb_fc: int = 0
        self._cb_t: float = time.time()
        self._inf_ms: float = 0.0
        self._history: deque = deque(maxlen=3600)
        self.active: bool = False
        self.inference_active: bool = False
        self._last_sub: float = 0.0
        self._npu: Optional[InferenceEngine] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────
    def start(self, npu_engine: InferenceEngine):
        self.decoder.start(self.source, self.mode)

        def _on_loop(loop_count):
            self.analytics.smooth_reset()
            self.analytics_2.smooth_reset()
            # Don't reset tracker — tracks persist across loop boundary

        self.decoder.on_loop_restart = _on_loop
        self._npu = npu_engine
        self.active = True
        t = threading.Thread(target=self._submit_loop, daemon=True, name=f"Cam-{self.cam_id}")
        t.start()
        log.info(f"[Cam-{self.cam_id}] '{self.name}' started (tracking={'ON' if self.tracking_enabled else 'OFF'})")

    def stop(self):
        self.active = False
        self.decoder.stop()
        log.info(f"[Cam-{self.cam_id}] stopped")

    def _submit_loop(self):
        while self.active:
            if not self.inference_active:
                time.sleep(0.05)
                continue
            frame = self.decoder.read()
            if frame is None:
                time.sleep(0.02)
                continue
            now = time.time()
            if now - self._last_sub < 1.0 / self.inf_fps_limit:
                time.sleep(0.005)
                continue
            self._last_sub = now
            self._npu.submit(self.cam_id, frame, self._on_inference, self.det_filter)

    # ── Inference callback ───────────────────────────────────────────────────
    def _on_inference(self, cam_id: int, frame: np.ndarray, raw_dets: list, inf_ms: float):
        h, w = frame.shape[:2]

        # ── Run ByteTrack if enabled ─────────────────────────────────────────
        if self.tracking_enabled and raw_dets:
            tracks = self.tracker.update(raw_dets)
            tracked_dets = []
            for t in tracks:
                tlbr = t.tlbr
                tracked_dets.append({
                    "bbox": [float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])],
                    "conf": t.score,
                    "cls": 0,
                    "track_id": t.track_id,
                })
            dets_for_analytics = tracked_dets
        else:
            dets_for_analytics = raw_dets

        # Pair 1 (Q1 + S1)
        if self.analytics.zones_defined():
            metrics = self.analytics.update(dets_for_analytics, w, h)
        else:
            metrics = dict(_EMPTY_METRICS, _tz={})

        # Pair 2 (Q2 + S2)
        if self.analytics_2.zones_defined():
            metrics_2 = self.analytics_2.update(dets_for_analytics, w, h)
        else:
            metrics_2 = dict(_EMPTY_METRICS, _tz={})

        tz = metrics.pop("_tz", {})
        tz2 = metrics_2.pop("_tz", {})

        # Merge pair-2 metrics with q2_ prefix
        for key in ("queue_length", "avg_waiting_time", "avg_processing_time",
                     "throughput_per_hour", "wait_method", "proc_method",
                     "queue_entries", "queue_exits", "service_exits",
                     "zone_counts", "open_counter", "crossing_log",
                     "total_processed", "L_queue_avg", "L_service_avg",
                     "lambda_exit_pm", "lambda_service_pm"):
            metrics[f"q2_{key}"] = metrics_2.get(key, 0)
        metrics["pair2_active"] = self.analytics_2.zones_defined()

        ann = self._draw(frame.copy(), dets_for_analytics, metrics, inf_ms, tz, tz2)

        self._cb_fc += 1
        now = time.time()
        if now - self._cb_t >= 1.0:
            self._inf_fps = self._cb_fc / (now - self._cb_t)
            self._cb_fc = 0
            self._cb_t = now
        self._inf_ms = inf_ms

        with self._lock:
            self._metrics = metrics
            self._ann = ann
            last_ts = self._history[-1].get("_ts", 0) if self._history else 0
            if now - last_ts >= 1.0:
                entry = {
                    "ts":              time.strftime("%H:%M:%S"),
                    "queue_length":    metrics["queue_length"],
                    "avg_wait":        metrics["avg_waiting_time"],
                    "wait_method":     metrics.get("wait_method", "none"),
                    "avg_proc":        metrics["avg_processing_time"],
                    "throughput":      metrics["throughput_per_hour"],
                    "in_service":      metrics.get("zone_counts", {}).get("service", 0),
                    "total_processed": metrics.get("total_processed", 0),
                    "queue_entries":   metrics.get("queue_entries", 0),
                    "queue_exits":     metrics.get("queue_exits", 0),
                    "open_counter":    metrics.get("open_counter", False),
                    "q2_queue_length": metrics.get("q2_queue_length", 0),
                    "q2_avg_wait":     metrics.get("q2_avg_waiting_time", 0),
                    "q2_avg_proc":     metrics.get("q2_avg_processing_time", 0),
                    "q2_throughput":   metrics.get("q2_throughput_per_hour", 0),
                    "tracking_mode":   metrics.get("tracking_mode", False),
                    "_ts":             now,
                }
                self._history.append(entry)
                if len(self._history) % 60 == 0:
                    self._persist_history()

    def _persist_history(self):
        try:
            path = ALERTS_DIR / f"cam_{self.cam_id}_history.json"
            data = [{k: v for k, v in e.items() if k != "_ts"} for e in self._history]
            with open(path, "w") as f:
                json.dump(data, f)
            log.info(f"[Cam-{self.cam_id}] History saved ({len(data)} samples)")
        except Exception as e:
            log.warning(f"[Cam-{self.cam_id}] history persist failed: {e}")

    # ── Drawing ──────────────────────────────────────────────────────────────
    def _draw(self, frame, dets, metrics, inf_ms, tz=None, tz2=None) -> np.ndarray:
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        if tz is None:
            tz = {}
        if tz2 is None:
            tz2 = {}

        diag = (w * w + h * h) ** 0.5
        draw_scale = diag / 800.0

        zone_thick = max(1, round(draw_scale * 1.5))
        lbl_scale = max(0.25, 0.38 * draw_scale)
        hud_scale = max(0.32, 0.42 * draw_scale)
        fill_alpha = max(0.05, 0.15 / draw_scale)
        bb_thick = max(1, round(draw_scale))
        id_scale = max(0.25, 0.32 * draw_scale)

        zone_colors_draw = {
            "queue": (0, 200, 255), "service": (0, 230, 100),
            "queue_2": (0, 165, 255), "service_2": (80, 255, 80),
        }

        # ── Zone polygons ────────────────────────────────────────────────────
        zone_draw_list = [
            ("queue",     self.analytics.zone_queue,     "Q1"),
            ("service",   self.analytics.zone_service,   "S1"),
            ("queue_2",   self.analytics_2.zone_queue,   "Q2"),
            ("service_2", self.analytics_2.zone_service, "S2"),
        ]
        for zone_key, poly, label in zone_draw_list:
            if poly and len(poly) >= 3:
                pts = np.array(
                    [[int(p[0] * w), int(p[1] * h)] for p in poly], np.int32,
                )
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], zone_colors_draw[zone_key])
                cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)
                cv2.polylines(frame, [pts], True, zone_colors_draw[zone_key], zone_thick, cv2.LINE_AA)
                cx = int(sum(p[0] for p in poly) / len(poly) * w)
                cy = int(sum(p[1] for p in poly) / len(poly) * h)
                cv2.putText(frame, label, (cx - 20, cy),
                            font, lbl_scale * 1.1, zone_colors_draw[zone_key],
                            max(1, round(draw_scale)), cv2.LINE_AA)

        # ── Bounding boxes + track IDs ───────────────────────────────────────
        for det in dets:
            bbox = det["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            bbox_key = (int(bbox[0]), int(bbox[1]))
            tid = det.get("track_id")

            # Determine zone for color
            zone = tz.get(bbox_key) or tz2.get(bbox_key)
            if zone == "queue":
                color = PERSON_C["queue"]
            elif zone == "service":
                color = PERSON_C["service"]
            elif tid is not None:
                color = _TRACK_PALETTE[tid % len(_TRACK_PALETTE)]
            else:
                color = PERSON_C[None]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, bb_thick, cv2.LINE_AA)

            # Track ID label
            if tid is not None:
                label = f"#{tid}"
                (tw, th), _ = cv2.getTextSize(label, font, id_scale, 1)
                cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 2),
                            font, id_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # ── HUD strip ────────────────────────────────────────────────────────
        wm = metrics.get("wait_method", "none")
        wlbl = ("T" if wm == "dwell_time" else
                "L" if wm == "littles_law" else
                "~" if wm == "estimated" else
                "W" if wm == "warming_up" else "\u2014")
        pm = metrics.get("proc_method", "none")
        plbl = ("T" if pm == "dwell_time" else
                "L" if pm == "littles_law" else
                "~" if pm == "estimated" else
                "W" if pm == "warming_up" else "\u2014")
        ql = metrics.get("queue_length", 0)
        sl = metrics.get("zone_counts", {}).get("service", 0)
        awt = metrics.get("avg_waiting_time", 0.0)
        apt = metrics.get("avg_processing_time", 0.0)
        qin = metrics.get("queue_entries", 0)
        qout = metrics.get("queue_exits", 0)
        tph = metrics.get("throughput_per_hour", 0.0)
        oc = "OPEN COUNTER" if metrics.get("open_counter") else ""
        trk = "TRK" if metrics.get("tracking_mode") else "DET"

        awt_str = f"{awt:.0f}s" if wm not in ("none", "no_data", "warming_up") else "\u2014"
        apt_str = f"{apt:.0f}s" if pm not in ("none", "no_data", "warming_up") else "\u2014"
        hud_text = (
            f"[{trk}] Q:{ql} S:{sl} | "
            f"Wait[{wlbl}]:{awt_str}  Proc[{plbl}]:{apt_str} | "
            f"In:{qin} Out:{qout} Tput:{tph:.0f}/h | "
            f"{self._inf_fps:.0f}fps {inf_ms:.0f}ms"
            + (f" | {oc}" if oc else "")
        )
        (tw, th_txt), _ = cv2.getTextSize(hud_text, font, hud_scale, 1)
        strip_h = th_txt + max(8, int(10 * draw_scale))

        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, strip_h + 4), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, hud_text, (6, strip_h), font, hud_scale, (220, 220, 220), 1, cv2.LINE_AA)

        # ── Alert banner ─────────────────────────────────────────────────────
        if metrics.get("alerts"):
            al = metrics["alerts"][0]
            msg = al["message"]
            bg = (0, 0, 160) if al.get("level") == "critical" else (0, 60, 120)
            (mw, mh), _ = cv2.getTextSize(msg, font, hud_scale, 1)
            bx = (w - mw) // 2
            by = h - 12
            cv2.rectangle(frame, (bx - 8, by - mh - 6), (bx + mw + 8, by + 4), bg, -1)
            cv2.putText(frame, msg, (bx, by), font, hud_scale, (255, 220, 120), 1, cv2.LINE_AA)

        return frame

    # ── Public API ───────────────────────────────────────────────────────────
    def get_metrics(self) -> Dict:
        with self._lock:
            return dict(self._metrics)

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            if self._ann is not None:
                ret, buf = cv2.imencode(".jpg", self._ann, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return buf.tobytes() if ret else None
        raw = self.decoder.read()
        if raw is None:
            return None
        ret, buf = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ret else None

    def get_history(self) -> List:
        with self._lock:
            return [{k: v for k, v in e.items() if k != "_ts"} for e in self._history]

    def get_history_csv(self) -> str:
        with self._lock:
            rows = [{k: v for k, v in e.items() if k != "_ts"} for e in self._history]
        if not rows:
            return "ts,queue_length,avg_wait,avg_proc,throughput,in_service,total_processed\n"
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()

    def set_zones(self, q, s, e, q2=None, s2=None):
        with self._lock:
            self.analytics.set_zones(q, s, e)
            self.analytics_2.set_zones(q2 or [], s2 or [], [])

    def set_lines(self, entry_line, exit_line, entry_dir=1, exit_dir=-1):
        with self._lock:
            self.analytics.set_lines(entry_line, exit_line, entry_dir, exit_dir)

    def set_thresholds(self, alert_thr, wait_thr):
        with self._lock:
            self.analytics.alert_thr = alert_thr
            self.analytics.wait_thr = wait_thr

    def set_det_filter(self, det_filter: dict):
        with self._lock:
            self.det_filter.update(det_filter)

    def reset(self):
        with self._lock:
            self.analytics.reset()
            self.analytics_2.reset()
            self.tracker.reset()

    def to_dict(self) -> Dict:
        return {
            "cam_id":             self.cam_id,
            "name":               self.name,
            "source":             self.source,
            "mode":               self.mode,
            "active":             self.active,
            "fps":                round(self.decoder.fps, 1),
            "inf_fps":            round(self._inf_fps, 1),
            "inf_fps_limit":      self.inf_fps_limit,
            "inf_ms":             round(self._inf_ms, 1),
            "decode_method":      self.decoder.decode_method,
            "zones_defined":      self.analytics.zones_defined(),
            "zones_defined_2":    self.analytics_2.zones_defined(),
            "queue_zone_ready":   bool(self.analytics.zone_queue),
            "service_zone_ready": bool(self.analytics.zone_service),
            "lines_defined":      self.analytics.lines_defined(),
            "exit_zone_defined":  self.analytics.exit_zone_defined(),
            "inference_active":   self.inference_active,
            "tracking_enabled":   self.tracking_enabled,
            "det_filter":         dict(self.det_filter),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Camera Manager
# ─────────────────────────────────────────────────────────────────────────────
class CameraManager:
    def __init__(self, npu_engine: InferenceEngine):
        self.npu = npu_engine
        self.cameras: Dict[int, CameraStream] = {}
        self._lock = threading.Lock()

    def add(self, name: str, source: str, mode: str = "live",
            alert_thr: int = 10, wait_thr: float = 300.0,
            inf_fps: float = 15.0) -> CameraStream:
        cam = CameraStream(name, source, mode, alert_thr, wait_thr, inf_fps)
        cam.start(self.npu)
        with self._lock:
            self.cameras[cam.cam_id] = cam
        log.info(f"[Manager] Added cam {cam.cam_id} '{name}'")
        return cam

    def remove(self, cam_id: int):
        with self._lock:
            cam = self.cameras.pop(cam_id, None)
        if cam:
            cam.stop()

    def get(self, cam_id: int) -> Optional[CameraStream]:
        with self._lock:
            return self.cameras.get(cam_id)

    def list(self) -> List[Dict]:
        with self._lock:
            return [c.to_dict() for c in self.cameras.values()]

    def stop_all(self):
        with self._lock:
            for c in self.cameras.values():
                c.stop()
            self.cameras.clear()
