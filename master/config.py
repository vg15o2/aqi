"""
Global configuration — paths, thresholds, logging, thread tuning.
"""

import os
import sys
import logging
from pathlib import Path

import cv2

# ── Thread tuning ────────────────────────────────────────────────────────────
os.environ.setdefault("OPENVINO_NUM_THREADS", "6")
os.environ.setdefault("OMP_NUM_THREADS", "6")
cv2.setNumThreads(4)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("queue_system.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("AirportQueue")

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
LOITER_DIR = SCRIPT_DIR.parent / "loitering_alerts"
ALERTS_DIR = SCRIPT_DIR / "queue_alerts"
ALERTS_DIR.mkdir(exist_ok=True)

# ── Model paths (searched in order) ──────────────────────────────────────────
MODEL_CANDIDATES = [
    SCRIPT_DIR / "yolo26n_int8_openvino_model" / "yolo26n.xml",        # YOLO26n (preferred)
    SCRIPT_DIR / "latest_int8_openvino_model"  / "latest.xml",
    Path("/home/gvaishnavi/qm/best_int8_openvino_model/best.xml"),     # existing model (fallback)
    # Add your model path here:
    # Path("/path/to/your/model.xml"),
]

# ── Inference backend ────────────────────────────────────────────────────────
# "axelera" | "openvino" | "auto"
# "auto" tries Axelera first, falls back to OpenVINO
INFERENCE_BACKEND = "openvino"   # "openvino" for CPU/GPU testing, "axelera" for edge, "auto" tries both

# ── Model architecture ───────────────────────────────────────────────────────
# "yolo26" — NMS-free, DFL-free. Output: [batch, max_det, 6] (x1,y1,x2,y2,conf,cls)
# "yolov8" — requires NMS. Output: [batch, 4+nc, anchors]
# "auto"   — detect from output shape (6 cols = yolo26, else yolov8)
MODEL_ARCH = "auto"

# ── Detection thresholds (global defaults, overridable per-camera) ───────────
CONF_THR   = 0.30
IOU_THR    = 0.30     # only used for yolov8 NMS fallback
INF_SIZE   = 640
MIN_BOX_PX = 15

# ── Per-camera detection filter defaults ─────────────────────────────────────
DEFAULT_DET_FILTER = {
    "conf_thr":      CONF_THR,
    "min_box_px":    MIN_BOX_PX,
    "min_box_area":  0,
    "max_box_area":  0,
    "hw_ratio_min":  0.0,
    "hw_ratio_max":  0.0,
}

# ── Tracker defaults ─────────────────────────────────────────────────────────
TRACKER_TRACK_THRESH  = 0.45
TRACKER_MATCH_THRESH  = 0.8
TRACKER_TRACK_BUFFER  = 30
TRACKER_LOW_THRESH    = 0.1
TRACKER_BBOX_INFLATE  = 2.0   # bbox doubling: inflate by 2x before tracking

log.info("=" * 68)
log.info("  AIRPORT QUEUE INTELLIGENCE — Axelera Metis + Intel iGPU")
log.info("=" * 68)
