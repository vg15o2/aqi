"""
Quick accuracy test — run model on video, draw zones, see detections + tracking.

Usage:
  python3 test_accuracy.py --source /path/to/video.mp4
  python3 test_accuracy.py --source rtsp://...

Controls:
  q       — quit
  SPACE   — pause/resume
  z       — enter zone drawing mode (click to draw polygon, right-click to finish)
  r       — reset zones
  +/-     — adjust confidence threshold
  t       — toggle tracking on/off
  b       — toggle bounding boxes on/off
  s       — save current frame as screenshot
"""

import argparse
import sys
import time

import cv2
import numpy as np

# ── Bootstrap (import from master modules) ───────────────────────────────────
from config import (
    MODEL_CANDIDATES, MODEL_ARCH, CONF_THR, INF_SIZE, MIN_BOX_PX,
    DEFAULT_DET_FILTER, TRACKER_BBOX_INFLATE, log,
)
from inference import (
    preprocess, postprocess_yolo26, postprocess_yolov8, OpenVINOBackend,
)
from bytetrack import ByteTracker
from zones import in_poly


# ── Zone drawing state ───────────────────────────────────────────────────────
_drawing_zone = None       # "queue" or "service" or None
_current_points = []       # points being drawn
_zone_queue = []           # normalized polygon [[x,y], ...]
_zone_service = []
_conf_thr = CONF_THR
_tracking_on = True
_show_boxes = True
_paused = False


def _mouse_cb(event, x, y, flags, param):
    global _current_points, _drawing_zone, _zone_queue, _zone_service
    fw, fh = param
    if _drawing_zone is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        _current_points.append([x / fw, y / fh])
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(_current_points) >= 3:
            if _drawing_zone == "queue":
                _zone_queue = list(_current_points)
                log.info(f"Queue zone set: {len(_zone_queue)} points")
            elif _drawing_zone == "service":
                _zone_service = list(_current_points)
                log.info(f"Service zone set: {len(_zone_service)} points")
        _current_points = []
        _drawing_zone = None


def _in_zone_simple(bbox, fw, fh, poly):
    """Check if bbox center-bottom is in polygon."""
    if not poly or len(poly) < 3:
        return False
    cx = ((bbox[0] + bbox[2]) / 2) / fw
    by = bbox[3] / fh
    return in_poly((cx, by), poly)


def main():
    global _drawing_zone, _current_points, _zone_queue, _zone_service
    global _conf_thr, _tracking_on, _show_boxes, _paused

    parser = argparse.ArgumentParser(description="Test detection accuracy")
    parser.add_argument("--source", required=True, help="Video file or RTSP URL")
    parser.add_argument("--conf", type=float, default=CONF_THR, help="Initial confidence threshold")
    parser.add_argument("--no-track", action="store_true", help="Disable tracking")
    args = parser.parse_args()

    _conf_thr = args.conf
    _tracking_on = not args.no_track

    # ── Load model ───────────────────────────────────────────────────────────
    model_path = None
    for p in MODEL_CANDIDATES:
        if p.exists():
            model_path = str(p)
            break
    if model_path is None:
        print("ERROR: No model found. Check MODEL_CANDIDATES in config.py")
        print(f"Searched: {[str(p) for p in MODEL_CANDIDATES]}")
        sys.exit(1)

    backend = OpenVINOBackend()
    if not backend.load(model_path):
        print("ERROR: Failed to load model")
        sys.exit(1)
    print(f"Model: {model_path}")
    print(f"Device: {backend.device_name()}")

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.source}")
        sys.exit(1)

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"Video: {fw}x{fh} @ {fps:.0f} FPS")

    tracker = ByteTracker(
        track_thresh=0.45,
        match_thresh=0.8,
        track_buffer=30,
        bbox_inflate=TRACKER_BBOX_INFLATE,
    )

    cv2.namedWindow("AQI Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AQI Test", min(fw, 1280), min(fh, 720))
    cv2.setMouseCallback("AQI Test", _mouse_cb, (fw, fh))

    palette = [
        (230, 100, 50), (50, 180, 230), (100, 230, 50), (230, 50, 180),
        (50, 230, 180), (180, 50, 230), (230, 180, 50), (50, 100, 230),
    ]

    frame_count = 0
    det_count = 0
    t_start = time.time()

    print("\nControls:")
    print("  q=quit  SPACE=pause  z=draw queue zone  x=draw service zone")
    print("  r=reset zones  +/-=conf threshold  t=toggle tracking  b=toggle boxes  s=screenshot\n")

    while True:
        if not _paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame_count += 1
        else:
            time.sleep(0.03)

        display = frame.copy()

        # ── Inference ────────────────────────────────────────────────────────
        if not _paused:
            blob, scale, pw, ph, ow, oh = preprocess(frame)
            out = backend.infer(blob)

            det_filter = dict(DEFAULT_DET_FILTER)
            det_filter["conf_thr"] = _conf_thr

            # Auto-detect arch
            arch = MODEL_ARCH
            if arch == "auto":
                arch = "yolo26" if (out.ndim == 3 and out.shape[-1] == 6) else "yolov8"

            if arch == "yolo26":
                dets = postprocess_yolo26(out, scale, pw, ph, ow, oh, det_filter)
            else:
                dets = postprocess_yolov8(out, scale, pw, ph, ow, oh, det_filter)

            det_count = len(dets)

            # ── Tracking ─────────────────────────────────────────────────────
            if _tracking_on and dets:
                tracks = tracker.update(dets)
                tracked_dets = []
                for t in tracks:
                    tracked_dets.append({
                        "bbox": [float(t.tlbr[0]), float(t.tlbr[1]),
                                 float(t.tlbr[2]), float(t.tlbr[3])],
                        "conf": t.score,
                        "cls": 0,
                        "track_id": t.track_id,
                    })
                draw_dets = tracked_dets
            elif _tracking_on and not dets:
                tracks = tracker.update([])
                draw_dets = []
            else:
                draw_dets = dets

        # ── Draw zones ───────────────────────────────────────────────────────
        q_count = 0
        s_count = 0
        for name, poly, color in [("QUEUE", _zone_queue, (0, 200, 255)),
                                    ("SERVICE", _zone_service, (0, 255, 128))]:
            if poly and len(poly) >= 3:
                pts = np.array([[int(p[0] * fw), int(p[1] * fh)] for p in poly], np.int32)
                ov = display.copy()
                cv2.fillPoly(ov, [pts], color)
                cv2.addWeighted(ov, 0.15, display, 0.85, 0, display)
                cv2.polylines(display, [pts], True, color, 2, cv2.LINE_AA)
                cx = int(sum(p[0] for p in poly) / len(poly) * fw)
                cy = int(sum(p[1] for p in poly) / len(poly) * fh)
                cv2.putText(display, name, (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # ── Draw current drawing points ──────────────────────────────────────
        if _drawing_zone and _current_points:
            color = (0, 200, 255) if _drawing_zone == "queue" else (0, 255, 128)
            for i, pt in enumerate(_current_points):
                px, py = int(pt[0] * fw), int(pt[1] * fh)
                cv2.circle(display, (px, py), 5, color, -1)
                if i > 0:
                    prev = _current_points[i - 1]
                    cv2.line(display, (int(prev[0] * fw), int(prev[1] * fh)),
                             (px, py), color, 2)

        # ── Draw detections ──────────────────────────────────────────────────
        if _show_boxes:
            for det in draw_dets:
                bbox = det["bbox"]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                tid = det.get("track_id")
                conf = det["conf"]

                # Zone color
                in_q = _in_zone_simple(bbox, fw, fh, _zone_queue)
                in_s = _in_zone_simple(bbox, fw, fh, _zone_service)
                if in_s:
                    color = (0, 255, 128)
                    s_count += 1
                elif in_q:
                    color = (0, 200, 255)
                    q_count += 1
                elif tid is not None:
                    color = palette[tid % len(palette)]
                else:
                    color = (180, 180, 180)

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                # Label
                if tid is not None:
                    label = f"#{tid} {conf:.2f}"
                else:
                    label = f"{conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(display, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
                cv2.putText(display, label, (x1 + 2, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # ── HUD ──────────────────────────────────────────────────────────────
        elapsed = time.time() - t_start
        proc_fps = frame_count / elapsed if elapsed > 0 else 0
        mode = "TRK" if _tracking_on else "DET"
        hud = (f"[{mode}] Dets:{det_count} Q:{q_count} S:{s_count} | "
               f"Conf:{_conf_thr:.2f} | {proc_fps:.1f} FPS"
               + (" | PAUSED" if _paused else ""))
        if _drawing_zone:
            hud += f" | DRAWING {_drawing_zone.upper()} (left-click=add point, right-click=finish)"

        cv2.rectangle(display, (0, 0), (fw, 28), (10, 10, 10), -1)
        cv2.putText(display, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imshow("AQI Test", display)

        # ── Key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            _paused = not _paused
        elif key == ord('z'):
            _drawing_zone = "queue"
            _current_points = []
            print("Drawing QUEUE zone — left-click points, right-click to finish")
        elif key == ord('x'):
            _drawing_zone = "service"
            _current_points = []
            print("Drawing SERVICE zone — left-click points, right-click to finish")
        elif key == ord('r'):
            _zone_queue = []
            _zone_service = []
            _current_points = []
            _drawing_zone = None
            print("Zones reset")
        elif key == ord('+') or key == ord('='):
            _conf_thr = min(0.95, _conf_thr + 0.05)
            print(f"Conf threshold: {_conf_thr:.2f}")
        elif key == ord('-'):
            _conf_thr = max(0.05, _conf_thr - 0.05)
            print(f"Conf threshold: {_conf_thr:.2f}")
        elif key == ord('t'):
            _tracking_on = not _tracking_on
            print(f"Tracking: {'ON' if _tracking_on else 'OFF'}")
        elif key == ord('b'):
            _show_boxes = not _show_boxes
            print(f"Bounding boxes: {'ON' if _show_boxes else 'OFF'}")
        elif key == ord('s'):
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, display)
            print(f"Saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
