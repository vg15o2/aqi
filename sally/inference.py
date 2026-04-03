"""
Inference — Ultralytics YOLO. Auto-downloads model on first run.
"""

from ultralytics import YOLO
from config import MODEL, CONF_THR, IOU_THR, INF_SIZE, PERSON_CLS


def load_model(model_path: str = MODEL):
    """Load YOLO model. Auto-downloads if not found locally."""
    model = YOLO(model_path)
    return model


def detect(model, frame, conf_thr=CONF_THR):
    """
    Run detection on a single frame.
    Returns list of {bbox: [x1,y1,x2,y2], conf: float, cls: int}
    """
    results = model.predict(
        frame,
        conf=conf_thr,
        iou=IOU_THR,
        imgsz=INF_SIZE,
        classes=[PERSON_CLS],
        verbose=False,
    )

    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append({
                "bbox": [x1, y1, x2, y2],
                "conf": float(box.conf[0]),
                "cls": int(box.cls[0]),
            })
    return dets
