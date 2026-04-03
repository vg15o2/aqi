"""
Inference Engine — abstract base with Axelera and OpenVINO backends.

Supports two model architectures:
  YOLO26  — NMS-free, DFL-free. Output: [batch, max_det, 6] (x1,y1,x2,y2,conf,cls).
             No NMS postprocessing needed.
  YOLOv8  — requires NMS. Output: [batch, 4+nc, anchors]. Legacy fallback.
"""

import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, List

import cv2
import numpy as np

from config import (
    CONF_THR, IOU_THR, INF_SIZE, MIN_BOX_PX, DEFAULT_DET_FILTER,
    MODEL_CANDIDATES, MODEL_ARCH, INFERENCE_BACKEND, SCRIPT_DIR, log,
)


# ─────────────────────────────────────────────────────────────────────────────
# NMS (only needed for YOLOv8 fallback)
# ─────────────────────────────────────────────────────────────────────────────
def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / u if u > 0 else 0.0


def nms(dets: list, thr: float) -> list:
    if not dets:
        return []
    dets = sorted(dets, key=lambda x: x["conf"], reverse=True)
    keep = []
    sup = [False] * len(dets)
    for i in range(len(dets)):
        if sup[i]:
            continue
        keep.append(dets[i])
        for j in range(i + 1, len(dets)):
            if not sup[j] and _iou(dets[i]["bbox"], dets[j]["bbox"]) > thr:
                sup[j] = True
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing (shared across backends)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(frame: np.ndarray, inf_size: int = INF_SIZE):
    """Letterbox resize preserving aspect ratio, pad to square."""
    h, w = frame.shape[:2]
    scale = min(inf_size / w, inf_size / h)
    nw, nh = int(w * scale), int(h * scale)
    pw, ph = (inf_size - nw) // 2, (inf_size - nh) // 2
    padded = np.full((inf_size, inf_size, 3), 114, np.uint8)
    padded[ph:ph + nh, pw:pw + nw] = cv2.resize(
        frame, (nw, nh), interpolation=cv2.INTER_LINEAR,
    )
    blob = (np.ascontiguousarray(padded.astype(np.float32) / 255.0)
            .transpose(2, 0, 1)[np.newaxis])
    return blob, scale, pw, ph, w, h


# ─────────────────────────────────────────────────────────────────────────────
# Postprocessing — YOLO26 (NMS-free)
# ─────────────────────────────────────────────────────────────────────────────
def postprocess_yolo26(out, scale, pw, ph, ow, oh, det_filter: dict = None):
    """
    YOLO26 NMS-free output: [batch, max_det, 6] — (x1, y1, x2, y2, conf, cls).
    Coordinates are in inference-space (640x640 with letterbox padding).
    """
    f = det_filter or DEFAULT_DET_FILTER
    conf_thr = f.get("conf_thr", CONF_THR)
    min_px   = f.get("min_box_px", MIN_BOX_PX)
    min_area = f.get("min_box_area", 0)
    max_area = f.get("max_box_area", 0)
    hw_min   = f.get("hw_ratio_min", 0.0)
    hw_max   = f.get("hw_ratio_max", 0.0)

    # Handle different output shapes
    if out.ndim == 3:
        out = out[0]  # remove batch dim → [max_det, 6]

    dets = []
    for row in out:
        if len(row) < 6:
            continue
        x1_inf, y1_inf, x2_inf, y2_inf, conf, cls_id = row[:6]
        conf = float(conf)
        cls_id = int(cls_id)

        if conf < conf_thr:
            continue
        if cls_id != 0:  # person class only
            continue

        # Map from inference space back to original frame
        x1 = max(0.0, (float(x1_inf) - pw) / scale)
        y1 = max(0.0, (float(y1_inf) - ph) / scale)
        x2 = min(ow - 1.0, (float(x2_inf) - pw) / scale)
        y2 = min(oh - 1.0, (float(y2_inf) - ph) / scale)

        bw_px = x2 - x1
        bh_px = y2 - y1
        if bw_px <= 0 or bh_px <= 0:
            continue
        if bw_px < min_px or bh_px < min_px:
            continue
        area = bw_px * bh_px
        if min_area > 0 and area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        hw = bh_px / (bw_px + 1e-6)
        if hw_min > 0 and hw < hw_min:
            continue
        if hw_max > 0 and hw > hw_max:
            continue

        dets.append({"bbox": [x1, y1, x2, y2], "conf": conf, "cls": 0})

    return dets  # no NMS needed — model handles deduplication


# ─────────────────────────────────────────────────────────────────────────────
# Postprocessing — YOLOv8 (legacy, requires NMS)
# ─────────────────────────────────────────────────────────────────────────────
def postprocess_yolov8(out, scale, pw, ph, ow, oh, det_filter: dict = None):
    """YOLOv8 raw output: [batch, 4+nc, anchors] — needs NMS."""
    f = det_filter or DEFAULT_DET_FILTER
    conf_thr = f.get("conf_thr", CONF_THR)
    min_px   = f.get("min_box_px", MIN_BOX_PX)
    min_area = f.get("min_box_area", 0)
    max_area = f.get("max_box_area", 0)
    hw_min   = f.get("hw_ratio_min", 0.0)
    hw_max   = f.get("hw_ratio_max", 0.0)

    if out.ndim == 3:
        out = out[0].T

    dets = []
    for row in out:
        cx, cy, bw, bh = row[:4]
        scores = row[4:]
        cls = int(np.argmax(scores))
        conf = float(scores[cls])
        if cls != 0 or conf < conf_thr:
            continue
        x1 = max(0.0, ((cx - bw / 2) - pw) / scale)
        y1 = max(0.0, ((cy - bh / 2) - ph) / scale)
        x2 = min(ow - 1.0, ((cx + bw / 2) - pw) / scale)
        y2 = min(oh - 1.0, ((cy + bh / 2) - ph) / scale)
        bw_px = x2 - x1
        bh_px = y2 - y1
        if bw_px <= 0 or bh_px <= 0:
            continue
        if bw_px < min_px or bh_px < min_px:
            continue
        area = bw_px * bh_px
        if min_area > 0 and area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        hw = bh_px / (bw_px + 1e-6)
        if hw_min > 0 and hw < hw_min:
            continue
        if hw_max > 0 and hw > hw_max:
            continue
        dets.append({"bbox": [x1, y1, x2, y2], "conf": conf, "cls": 0})
    return nms(dets, IOU_THR)


def _postprocess(out, scale, pw, ph, ow, oh, det_filter=None, arch=MODEL_ARCH):
    """Route to correct postprocessor based on model architecture."""
    if arch == "auto":
        # Auto-detect: YOLO26 output is [batch, N, 6], YOLOv8 is [batch, 4+nc, anchors]
        if out.ndim == 3 and out.shape[-1] == 6:
            arch = "yolo26"
        elif out.ndim == 3 and out.shape[-1] > 6:
            arch = "yolo26"  # also [N, 6+] shaped
        else:
            arch = "yolov8"
    if arch == "yolo26":
        return postprocess_yolo26(out, scale, pw, ph, ow, oh, det_filter)
    return postprocess_yolov8(out, scale, pw, ph, ow, oh, det_filter)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Inference Backend
# ─────────────────────────────────────────────────────────────────────────────
class InferenceBackend(ABC):
    @abstractmethod
    def load(self, model_path: str) -> bool:
        """Load model. Return True on success."""

    @abstractmethod
    def infer(self, blob: np.ndarray) -> np.ndarray:
        """Run inference on preprocessed blob. Return raw output tensor."""

    @abstractmethod
    def device_name(self) -> str:
        """Human-readable device string."""


# ─────────────────────────────────────────────────────────────────────────────
# Axelera Voyager SDK Backend
# ─────────────────────────────────────────────────────────────────────────────
class AxeleraBackend(InferenceBackend):
    """
    Axelera Metis AIPU via Voyager SDK.
    Requires: voyager SDK installed, Metis M.2/PCIe hardware present.
    """

    def __init__(self):
        self._model = None
        self._device = "Axelera"

    def load(self, model_path: str) -> bool:
        try:
            from voyager import Voyager
            self._model = Voyager(model_path)
            log.info(f"[Axelera] Model loaded: {model_path}")
            return True
        except ImportError:
            log.warning("[Axelera] Voyager SDK not installed")
            return False
        except Exception as e:
            log.warning(f"[Axelera] Failed to load: {e}")
            return False

    def infer(self, blob: np.ndarray) -> np.ndarray:
        return self._model.run(blob)

    def device_name(self) -> str:
        return self._device


# ─────────────────────────────────────────────────────────────────────────────
# OpenVINO Backend (fallback)
# ─────────────────────────────────────────────────────────────────────────────
class OpenVINOBackend(InferenceBackend):
    def __init__(self):
        self._infer_req = None
        self._device = "CPU"

    def load(self, model_path: str) -> bool:
        try:
            from openvino import Core
            core = Core()
            avail = core.available_devices
            log.info(f"[OV] Available devices: {avail}")
            preferred = [d for d in ("NPU", "GPU", "CPU") if d in avail]
            dev = ("AUTO:" + ",".join(preferred)
                   if len(preferred) > 1
                   else (preferred[0] if preferred else "CPU"))
            config = {
                "PERFORMANCE_HINT": "THROUGHPUT",
                "CACHE_DIR": str(SCRIPT_DIR / ".ov_cache"),
            }
            model = core.read_model(model_path)
            cm = core.compile_model(model, dev, config)
            self._infer_req = cm.create_infer_request()
            self._device = dev
            log.info(f"[OV] Model loaded on {dev}: {model_path}")
            return True
        except Exception as e:
            log.error(f"[OV] Failed to load: {e}")
            return False

    def infer(self, blob: np.ndarray) -> np.ndarray:
        self._infer_req.infer({0: blob})
        return self._infer_req.get_output_tensor(0).data

    def device_name(self) -> str:
        return self._device


# ─────────────────────────────────────────────────────────────────────────────
# Inference Engine (worker pool wrapping a backend)
# ─────────────────────────────────────────────────────────────────────────────
class InferenceEngine:
    NUM_WORKERS = 2   # keep low for dev machines; increase on edge (8)

    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=128)
        self.workers: List[threading.Thread] = []
        self.running = False
        self.actual_dev = "none"
        self._lock = threading.Lock()
        self.stats = {"batches": 0, "dropped": 0, "total_det": 0, "inf_ms_sum": 0.0}
        self._backend_class = None
        self._model_path = None

    def start(self):
        if self.running:
            return
        self.running = True

        # Find model
        model_path = next((str(p) for p in MODEL_CANDIDATES if p.exists()), None)
        if model_path is None:
            log.error("[Engine] No model found — inference disabled")
            self.running = False
            return
        self._model_path = model_path

        # Select backend
        backend_order = []
        if INFERENCE_BACKEND in ("axelera", "auto"):
            backend_order.append(AxeleraBackend)
        if INFERENCE_BACKEND in ("openvino", "auto"):
            backend_order.append(OpenVINOBackend)

        # Probe which backend works
        self._backend_class = None
        for cls in backend_order:
            test = cls()
            if test.load(model_path):
                self._backend_class = cls
                self.actual_dev = test.device_name()
                log.info(f"[Engine] Using {cls.__name__} on {self.actual_dev}")
                break

        if self._backend_class is None:
            log.error("[Engine] No backend available — inference disabled")
            self.running = False
            return

        # Start workers (each creates its own backend instance)
        for i in range(self.NUM_WORKERS):
            t = threading.Thread(
                target=self._worker, args=(i,),
                daemon=True, name=f"InfWorker-{i}",
            )
            t.start()
            self.workers.append(t)
        log.info(f"[Engine] {self.NUM_WORKERS} workers on {self.actual_dev}")

    def stop(self):
        self.running = False
        for _ in self.workers:
            try:
                self.frame_queue.put_nowait(None)
            except Exception:
                pass
        for w in self.workers:
            w.join(timeout=2)
        self.workers.clear()

    def submit(self, cam_id: int, frame: np.ndarray, callback: Callable,
               det_filter: dict = None):
        if not self.running:
            return
        try:
            self.frame_queue.put_nowait(
                (cam_id, frame, callback, time.perf_counter(), det_filter)
            )
        except queue.Full:
            with self._lock:
                self.stats["dropped"] += 1

    def _worker(self, wid: int):
        # Each worker gets its own backend instance
        backend = self._backend_class()
        if not backend.load(self._model_path):
            log.error(f"[InfWorker-{wid}] Backend load failed")
            return

        while self.running:
            try:
                item = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            cam_id, frame, callback, t0, det_filter = item
            try:
                blob, scale, pw, ph, ow, oh = preprocess(frame)
                out = backend.infer(blob)
                dets = _postprocess(out, scale, pw, ph, ow, oh, det_filter)
                inf_ms = (time.perf_counter() - t0) * 1000
                with self._lock:
                    self.stats["batches"] += 1
                    self.stats["inf_ms_sum"] += inf_ms
                    self.stats["total_det"] += len(dets)
                if callback:
                    callback(cam_id, frame, dets, inf_ms)
            except Exception as e:
                log.error(f"[InfWorker-{wid}] cam {cam_id}: {e}")

    def get_stats(self) -> dict:
        with self._lock:
            b = max(1, self.stats["batches"])
            return {
                "device": self.actual_dev,
                "batches": self.stats["batches"],
                "avg_inf_ms": round(self.stats["inf_ms_sum"] / b, 1),
                "dropped": self.stats["dropped"],
            }
