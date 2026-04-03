"""
Inference — load OpenVINO model, preprocess, postprocess.
Supports YOLO26 (NMS-free) and YOLOv8 (with NMS).
"""

import numpy as np
import cv2

from config import CONF_THR, IOU_THR, INF_SIZE, MIN_BOX_PX, HW_RATIO_MIN, HW_RATIO_MAX


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str):
    """Load OpenVINO model. Returns (infer_request, device_name)."""
    from openvino import Core
    core = Core()
    avail = core.available_devices
    preferred = [d for d in ("GPU", "NPU", "CPU") if d in avail]
    dev = ("AUTO:" + ",".join(preferred)
           if len(preferred) > 1
           else (preferred[0] if preferred else "CPU"))
    config = {"PERFORMANCE_HINT": "LATENCY"}
    model = core.read_model(model_path)
    compiled = core.compile_model(model, dev, config)
    infer_req = compiled.create_infer_request()
    return infer_req, dev


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(frame: np.ndarray):
    """Letterbox resize to INF_SIZE square. Returns (blob, scale, pw, ph, ow, oh)."""
    h, w = frame.shape[:2]
    scale = min(INF_SIZE / w, INF_SIZE / h)
    nw, nh = int(w * scale), int(h * scale)
    pw, ph = (INF_SIZE - nw) // 2, (INF_SIZE - nh) // 2
    padded = np.full((INF_SIZE, INF_SIZE, 3), 114, np.uint8)
    padded[ph:ph + nh, pw:pw + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    blob = np.ascontiguousarray(padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
    return blob, scale, pw, ph, w, h


# ─────────────────────────────────────────────────────────────────────────────
# Postprocessing
# ─────────────────────────────────────────────────────────────────────────────
def _filter(x1, y1, x2, y2, conf, conf_thr):
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return False
    if conf < conf_thr:
        return False
    if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
        return False
    hw = bh / (bw + 1e-6)
    if HW_RATIO_MIN > 0 and hw < HW_RATIO_MIN:
        return False
    if HW_RATIO_MAX > 0 and hw > HW_RATIO_MAX:
        return False
    return True


def postprocess_yolo26(out, scale, pw, ph, ow, oh, conf_thr=CONF_THR):
    """YOLO26 NMS-free: [batch, N, 6] → list of {bbox, conf, cls}."""
    if out.ndim == 3:
        out = out[0]
    dets = []
    for row in out:
        if len(row) < 6:
            continue
        x1i, y1i, x2i, y2i, conf, cls_id = row[:6]
        if int(cls_id) != 0:
            continue
        x1 = max(0.0, (float(x1i) - pw) / scale)
        y1 = max(0.0, (float(y1i) - ph) / scale)
        x2 = min(ow - 1.0, (float(x2i) - pw) / scale)
        y2 = min(oh - 1.0, (float(y2i) - ph) / scale)
        if _filter(x1, y1, x2, y2, float(conf), conf_thr):
            dets.append({"bbox": [x1, y1, x2, y2], "conf": float(conf), "cls": 0})
    return dets


def postprocess_yolov8(out, scale, pw, ph, ow, oh, conf_thr=CONF_THR):
    """YOLOv8 raw: [batch, 4+nc, anchors] → NMS → list of {bbox, conf, cls}."""
    if out.ndim == 3:
        out = out[0].T
    dets = []
    for row in out:
        cx, cy, bw, bh = row[:4]
        scores = row[4:]
        cls = int(np.argmax(scores))
        conf = float(scores[cls])
        if cls != 0:
            continue
        x1 = max(0.0, ((cx - bw / 2) - pw) / scale)
        y1 = max(0.0, ((cy - bh / 2) - ph) / scale)
        x2 = min(ow - 1.0, ((cx + bw / 2) - pw) / scale)
        y2 = min(oh - 1.0, ((cy + bh / 2) - ph) / scale)
        if _filter(x1, y1, x2, y2, conf, conf_thr):
            dets.append({"bbox": [x1, y1, x2, y2], "conf": conf, "cls": 0})
    # NMS
    if not dets:
        return []
    dets.sort(key=lambda d: d["conf"], reverse=True)
    keep = []
    alive = [True] * len(dets)
    for i in range(len(dets)):
        if not alive[i]:
            continue
        keep.append(dets[i])
        for j in range(i + 1, len(dets)):
            if alive[j]:
                a, b = dets[i]["bbox"], dets[j]["bbox"]
                iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
                inter = iw * ih
                u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
                if u > 0 and inter / u > IOU_THR:
                    alive[j] = False
    return keep


def postprocess(out, scale, pw, ph, ow, oh, conf_thr=CONF_THR, arch="auto"):
    """Route to correct postprocessor."""
    if arch == "auto":
        arch = "yolo26" if (out.ndim == 3 and out.shape[-1] == 6) else "yolov8"
    if arch == "yolo26":
        return postprocess_yolo26(out, scale, pw, ph, ow, oh, conf_thr)
    return postprocess_yolov8(out, scale, pw, ph, ow, oh, conf_thr)
