"""
Configuration — model, thresholds, tracker settings.
"""

from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# ── Model ────────────────────────────────────────────────────────────────────
# Ultralytics model name or path to .pt file.
# "yolo11s.pt" auto-downloads from Ultralytics hub on first run.
MODEL = "yolo11s.pt"

# ── Detection ────────────────────────────────────────────────────────────────
CONF_THR    = 0.30
IOU_THR     = 0.45
INF_SIZE    = 640
PERSON_CLS  = 0       # COCO class 0 = person

# ── Tracker ──────────────────────────────────────────────────────────────────
TRACK_THRESH  = 0.45
MATCH_THRESH  = 0.8
TRACK_BUFFER  = 30
LOW_THRESH    = 0.1
BBOX_INFLATE  = 2.0   # bbox doubling (1.0 = off)
