"""
Airport Queue Intelligence — run detection, tracking, zone counting, dwell time.

Usage:
  python run.py --source video.mp4 --zones zones.json                   # run with saved zones
  python run.py --source video.mp4 --zones zones.json --draw-zones      # draw zones interactively
  python run.py --source rtsp://cam/stream --zones zones.json            # RTSP live

Controls (when --draw-zones):
  z           draw QUEUE zone (left-click = add point, right-click = finish)
  x           draw SERVICE zone
  r           reset all zones

Controls (always):
  +/-         adjust confidence threshold
  t           toggle tracking on/off
  b           toggle bounding boxes
  SPACE       pause/resume
  s           save screenshot
  q           quit (saves zones if --draw-zones)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np

from config import (
    MODEL_CANDIDATES, MODEL_ARCH, CONF_THR, INF_SIZE,
    TRACK_THRESH, MATCH_THRESH, TRACK_BUFFER, LOW_THRESH, BBOX_INFLATE,
)
from inference import load_model, preprocess, postprocess
from bytetrack import ByteTracker


# ─────────────────────────────────────────────────────────────────────────────
# Point-in-polygon (ray casting)
# ─────────────────────────────────────────────────────────────────────────────
def _in_poly(pt, poly):
    x, y = pt
    inside = False
    px, py = poly[-1]
    for nx, ny in poly:
        if ((ny > y) != (py > y)) and x < (px - nx) * (y - ny) / (py - ny + 1e-10) + nx:
            inside = not inside
        px, py = nx, ny
    return inside


def _bbox_in_zone(bbox, fw, fh, poly):
    """Check if bottom-center of bbox is inside normalized polygon."""
    if not poly or len(poly) < 3:
        return False
    cx = ((bbox[0] + bbox[2]) / 2) / fw
    by = bbox[3] / fh
    return _in_poly((cx, by), poly)


# ─────────────────────────────────────────────────────────────────────────────
# Dwell time tracker
# ─────────────────────────────────────────────────────────────────────────────
class DwellTracker:
    """Track per-person dwell time in zones. Compute avg wait, processing, throughput."""

    def __init__(self):
        self.zone_of: dict = {}            # track_id -> "queue" | "service"
        self.enter_time: dict = {}         # track_id -> timestamp
        self.active_q: set = set()
        self.active_s: set = set()
        self.queue_dwells: deque = deque(maxlen=500)    # (exit_ts, dwell_s)
        self.service_dwells: deque = deque(maxlen=500)
        self.queue_exits = 0
        self.service_exits = 0
        self.queue_entries = 0

    def update(self, tracked_dets, fw, fh, zone_q, zone_s):
        """Process one frame. Returns metrics dict."""
        now = time.time()
        cur_q, cur_s = set(), set()
        tz = {}  # bbox_key -> zone

        for det in tracked_dets:
            bbox = det["bbox"]
            tid = det.get("track_id")
            bk = (int(bbox[0]), int(bbox[1]))

            if _bbox_in_zone(bbox, fw, fh, zone_s):
                tz[bk] = "service"
                if tid is not None:
                    cur_s.add(tid)
                    if tid not in self.zone_of or self.zone_of[tid] != "service":
                        self.enter_time[tid] = now
                        self.zone_of[tid] = "service"
            elif _bbox_in_zone(bbox, fw, fh, zone_q):
                tz[bk] = "queue"
                if tid is not None:
                    cur_q.add(tid)
                    if tid not in self.zone_of or self.zone_of[tid] != "queue":
                        self.enter_time[tid] = now
                        self.zone_of[tid] = "queue"

        # Queue exits
        for tid in (self.active_q - cur_q):
            t0 = self.enter_time.pop(tid, None)
            if t0 and now - t0 > 0.5:
                self.queue_dwells.append((now, now - t0))
                self.queue_exits += 1
            self.zone_of.pop(tid, None)

        # Service exits
        for tid in (self.active_s - cur_s):
            t0 = self.enter_time.pop(tid, None)
            if t0 and now - t0 > 0.5:
                self.service_dwells.append((now, now - t0))
                self.service_exits += 1
            self.zone_of.pop(tid, None)

        # Queue entries
        new_q = cur_q - self.active_q
        self.queue_entries += len(new_q)

        self.active_q = cur_q
        self.active_s = cur_s

        # ── Compute metrics ──────────────────────────────────────────────────
        # Avg wait = mean dwell of people who completed their queue wait
        recent_q = [d for ts, d in self.queue_dwells if now - ts < 300]
        recent_s = [d for ts, d in self.service_dwells if now - ts < 300]

        avg_wait = round(sum(recent_q) / len(recent_q), 1) if recent_q else 0.0
        avg_proc = round(sum(recent_s) / len(recent_s), 1) if recent_s else 0.0

        # Throughput = service exits per hour (last 5 min window)
        svc_in_window = sum(1 for ts, _ in self.service_dwells if now - ts < 300)
        throughput_hr = round(svc_in_window * (3600 / 300), 1) if svc_in_window else 0.0

        # Little's Law fallback: if no completed dwells yet but people in queue
        if avg_wait == 0 and len(cur_q) > 0 and self.queue_exits > 0:
            # Estimate from current partial waits
            partials = [now - self.enter_time[tid] for tid in cur_q if tid in self.enter_time]
            if partials:
                avg_wait = round(sum(partials) / len(partials), 1)

        return {
            "q_count": len(cur_q),
            "s_count": len(cur_s),
            "avg_wait": avg_wait,
            "avg_proc": avg_proc,
            "throughput_hr": throughput_hr,
            "q_exits": self.queue_exits,
            "s_exits": self.service_exits,
            "q_entries": self.queue_entries,
            "tz": tz,
        }

    def reset(self):
        self.__init__()


# ─────────────────────────────────────────────────────────────────────────────
# Zone drawing state
# ─────────────────────────────────────────────────────────────────────────────
_draw_target = None      # "queue" or "service"
_draw_pts = []           # current polygon being drawn
_zone_q = []             # [[x_norm, y_norm], ...]
_zone_s = []


def _mouse_cb(event, x, y, flags, param):
    global _draw_pts, _draw_target, _zone_q, _zone_s
    fw, fh = param
    if _draw_target is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        _draw_pts.append([x / fw, y / fh])
    elif event == cv2.EVENT_RBUTTONDOWN and len(_draw_pts) >= 3:
        if _draw_target == "queue":
            _zone_q = list(_draw_pts)
            print(f"  Queue zone saved ({len(_zone_q)} points)")
        elif _draw_target == "service":
            _zone_s = list(_draw_pts)
            print(f"  Service zone saved ({len(_zone_s)} points)")
        _draw_pts = []
        _draw_target = None


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
_PALETTE = [
    (230,100,50),(50,180,230),(100,230,50),(230,50,180),
    (50,230,180),(180,50,230),(230,180,50),(50,100,230),
]
_Q_COLOR = (0, 200, 255)
_S_COLOR = (0, 255, 128)


def _draw_zone(frame, poly, color, label, fw, fh):
    if not poly or len(poly) < 3:
        return
    pts = np.array([[int(p[0]*fw), int(p[1]*fh)] for p in poly], np.int32)
    ov = frame.copy()
    cv2.fillPoly(ov, [pts], color)
    cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
    cx = int(sum(p[0] for p in poly) / len(poly) * fw)
    cy = int(sum(p[1] for p in poly) / len(poly) * fh)
    cv2.putText(frame, label, (cx-25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global _draw_target, _draw_pts, _zone_q, _zone_s

    parser = argparse.ArgumentParser(description="Airport Queue Intelligence")
    parser.add_argument("--source", required=True, help="Video file or RTSP URL")
    parser.add_argument("--zones", required=True, help="Path to zones JSON file")
    parser.add_argument("--draw-zones", action="store_true", help="Interactive zone drawing mode")
    parser.add_argument("--conf", type=float, default=CONF_THR, help="Confidence threshold")
    parser.add_argument("--model", type=str, default=None, help="Override model path (.xml)")
    parser.add_argument("--no-track", action="store_true", help="Disable tracking")
    args = parser.parse_args()

    conf_thr = args.conf
    tracking_on = not args.no_track
    show_boxes = True
    paused = False

    # ── Load zones ───────────────────────────────────────────────────────────
    zones_path = Path(args.zones)
    if zones_path.exists() and not args.draw_zones:
        with open(zones_path) as f:
            zdata = json.load(f)
        _zone_q = zdata.get("queue", [])
        _zone_s = zdata.get("service", [])
        print(f"Zones loaded from {zones_path}")
        if _zone_q:
            print(f"  Queue: {len(_zone_q)} points")
        if _zone_s:
            print(f"  Service: {len(_zone_s)} points")
    elif zones_path.exists() and args.draw_zones:
        # Load existing zones but allow editing
        with open(zones_path) as f:
            zdata = json.load(f)
        _zone_q = zdata.get("queue", [])
        _zone_s = zdata.get("service", [])
        print(f"Zones loaded for editing from {zones_path}")
    else:
        if not args.draw_zones:
            print(f"WARNING: {zones_path} not found. Use --draw-zones to create it.")

    # ── Load model ───────────────────────────────────────────────────────────
    model_path = args.model
    if model_path is None:
        model_path = next((str(p) for p in MODEL_CANDIDATES if p.exists()), None)
    if model_path is None:
        print("ERROR: No model found. Set model path in config.py or use --model")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    infer_req, dev = load_model(model_path)
    print(f"Device: {dev}")

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.source}")
        sys.exit(1)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {fw}x{fh}")

    # ── Init tracker + dwell tracker ─────────────────────────────────────────
    tracker = ByteTracker(
        track_thresh=TRACK_THRESH, match_thresh=MATCH_THRESH,
        track_buffer=TRACK_BUFFER, low_thresh=LOW_THRESH,
        bbox_inflate=BBOX_INFLATE,
    )
    dwell = DwellTracker()

    # ── Window ───────────────────────────────────────────────────────────────
    win = "AQI"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(fw, 1280), min(fh, 720))
    cv2.setMouseCallback(win, _mouse_cb, (fw, fh))

    frame_n = 0
    t_start = time.time()
    frame = None

    print("\nControls: q=quit  SPACE=pause  z=queue zone  x=service zone  r=reset")
    print("          +/-=conf  t=tracking  b=boxes  s=screenshot\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracker.reset()
                dwell.reset()
                continue
            frame_n += 1

        if frame is None:
            continue

        display = frame.copy()

        # ── Inference ────────────────────────────────────────────────────────
        if not paused:
            blob, sc, pw, ph, ow, oh = preprocess(frame)
            infer_req.infer({0: blob})
            out = infer_req.get_output_tensor(0).data
            dets = postprocess(out, sc, pw, ph, ow, oh, conf_thr, MODEL_ARCH)

            # Track
            if tracking_on and dets:
                tracks = tracker.update(dets)
                draw_dets = [{"bbox": [float(t.tlbr[0]), float(t.tlbr[1]),
                                       float(t.tlbr[2]), float(t.tlbr[3])],
                              "conf": t.score, "cls": 0, "track_id": t.track_id}
                             for t in tracks]
            elif tracking_on:
                tracker.update([])
                draw_dets = []
            else:
                draw_dets = dets

            # Dwell time
            metrics = dwell.update(draw_dets, fw, fh, _zone_q, _zone_s)
            tz = metrics["tz"]
        else:
            time.sleep(0.03)
            metrics = {"q_count": 0, "s_count": 0, "avg_wait": 0, "avg_proc": 0,
                       "throughput_hr": 0, "q_exits": 0, "s_exits": 0, "q_entries": 0}
            tz = {}
            draw_dets = []

        # ── Draw zones ───────────────────────────────────────────────────────
        _draw_zone(display, _zone_q, _Q_COLOR, "QUEUE", fw, fh)
        _draw_zone(display, _zone_s, _S_COLOR, "SERVICE", fw, fh)

        # Draw in-progress polygon
        if _draw_target and _draw_pts:
            c = _Q_COLOR if _draw_target == "queue" else _S_COLOR
            for i, pt in enumerate(_draw_pts):
                px, py = int(pt[0]*fw), int(pt[1]*fh)
                cv2.circle(display, (px, py), 5, c, -1)
                if i > 0:
                    pp = _draw_pts[i-1]
                    cv2.line(display, (int(pp[0]*fw), int(pp[1]*fh)), (px, py), c, 2)

        # ── Draw detections ──────────────────────────────────────────────────
        if show_boxes:
            for det in draw_dets:
                bbox = det["bbox"]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                tid = det.get("track_id")
                bk = (int(bbox[0]), int(bbox[1]))
                zone = tz.get(bk)

                color = (_S_COLOR if zone == "service" else
                         _Q_COLOR if zone == "queue" else
                         _PALETTE[tid % len(_PALETTE)] if tid else (180, 180, 180))

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                if tid is not None:
                    lbl = f"#{tid}"
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(display, (x1, y1-th-4), (x1+tw+4, y1), color, -1)
                    cv2.putText(display, lbl, (x1+2, y1-2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

        # ── HUD ──────────────────────────────────────────────────────────────
        fps = frame_n / (time.time() - t_start) if time.time() > t_start else 0
        mode = "TRK" if tracking_on else "DET"
        qc, sc = metrics["q_count"], metrics["s_count"]
        aw = metrics["avg_wait"]
        ap = metrics["avg_proc"]
        tph = metrics["throughput_hr"]
        qx, sx = metrics["q_exits"], metrics["s_exits"]

        aw_s = f"{aw:.0f}s" if aw > 0 else "--"
        ap_s = f"{ap:.0f}s" if ap > 0 else "--"
        hud = (f"[{mode}] Q:{qc} S:{sc} | Wait:{aw_s} Proc:{ap_s} | "
               f"Tput:{tph:.0f}/h | Exits Q:{qx} S:{sx} | "
               f"Conf:{conf_thr:.2f} {fps:.1f}fps")
        if paused:
            hud += " | PAUSED"
        if _draw_target:
            hud += f" | DRAWING {_draw_target.upper()}"

        cv2.rectangle(display, (0, 0), (fw, 28), (10, 10, 10), -1)
        cv2.putText(display, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imshow(win, display)

        # ── Keys ─────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('z'):
            _draw_target = "queue"; _draw_pts = []
            print("Draw QUEUE zone: left-click points, right-click to finish")
        elif key == ord('x'):
            _draw_target = "service"; _draw_pts = []
            print("Draw SERVICE zone: left-click points, right-click to finish")
        elif key == ord('r'):
            _zone_q = []; _zone_s = []; _draw_pts = []; _draw_target = None
            dwell.reset()
            print("Zones reset")
        elif key in (ord('+'), ord('=')):
            conf_thr = min(0.95, conf_thr + 0.05)
            print(f"Conf: {conf_thr:.2f}")
        elif key == ord('-'):
            conf_thr = max(0.05, conf_thr - 0.05)
            print(f"Conf: {conf_thr:.2f}")
        elif key == ord('t'):
            tracking_on = not tracking_on
            print(f"Tracking: {'ON' if tracking_on else 'OFF'}")
        elif key == ord('b'):
            show_boxes = not show_boxes
        elif key == ord('s'):
            fn = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fn, display)
            print(f"Saved: {fn}")

    # ── Save zones on quit ───────────────────────────────────────────────────
    if args.draw_zones and (_zone_q or _zone_s):
        zdata = {}
        if _zone_q:
            zdata["queue"] = _zone_q
        if _zone_s:
            zdata["service"] = _zone_s
        with open(zones_path, "w") as f:
            json.dump(zdata, f, indent=2)
        print(f"Zones saved to {zones_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
