"""
Configuration — model path, thresholds, tracker settings.
Edit this file to point to your model and tune detection.
"""

from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# ── Model ────────────────────────────────────────────────────────────────────
# First existing path wins. Add your model path here.
MODEL_CANDIDATES = [
    SCRIPT_DIR / "yolo26n_int8_openvino_model" / "yolo26n.xml",
    SCRIPT_DIR / "best_int8_openvino_model" / "best.xml",
    Path("/home/gvaishnavi/qm/best_int8_openvino_model/best.xml"),
    # Path("C:/Users/you/models/yolo26n_openvino/yolo26n.xml"),  # Windows example
]

# "yolo26" = NMS-free output [batch, N, 6] (x1,y1,x2,y2,conf,cls)
# "yolov8" = raw output needing NMS [batch, 4+nc, anchors]
# "auto"   = detect from output shape
MODEL_ARCH = "auto"

# ── Detection ────────────────────────────────────────────────────────────────
CONF_THR    = 0.30    # confidence threshold (tune with +/- at runtime)
IOU_THR     = 0.30    # NMS IoU threshold (only for yolov8)
INF_SIZE    = 640     # inference resolution (640 or 960)
MIN_BOX_PX  = 15      # reject tiny detections
HW_RATIO_MIN = 0.0    # min height/width ratio (1.0 = filter square objects like stanchions)
HW_RATIO_MAX = 0.0    # max height/width ratio (0 = disabled)

# ── Tracker ──────────────────────────────────────────────────────────────────
TRACK_THRESH  = 0.45   # high-conf for stage-1 association
MATCH_THRESH  = 0.8    # IoU cost threshold
TRACK_BUFFER  = 30     # frames to keep lost tracks
LOW_THRESH    = 0.1    # low-conf for stage-2 association
BBOX_INFLATE  = 2.0    # bbox doubling (2.0 = double size for tracking IoU, 1.0 = off)
