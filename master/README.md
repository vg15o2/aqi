# Airport Queue Intelligence — Edge CV Pipeline

Detect people in airport queues from CCTV feeds, track them across frames, count queue/service zone occupancy, and estimate wait times and processing times.

## Architecture

```
Video/RTSP → Decode (iGPU/CPU) → YOLO Detection (GPU/NPU/Axelera)
  → ByteTrack Tracking (CPU) → Zone Analytics (CPU) → Metrics
```

### Modules

| File | What |
|---|---|
| `test_accuracy.py` | **Start here.** Interactive test — run model on video, draw zones, see detections. |
| `config.py` | All configuration — model path, thresholds, tracker settings |
| `inference.py` | Inference engine — supports OpenVINO (CPU/GPU/NPU) and Axelera backends. YOLO26 (NMS-free) + YOLOv8 postprocessing. |
| `bytetrack.py` | Multi-object tracker with bounding box doubling for small overhead targets |
| `analytics.py` | Queue analytics — dwell-time tracking (with tracker) + Little's Law fallback (without) |
| `zones.py` | Point-in-polygon geometry |
| `decoder.py` | Video decode with hardware acceleration fallback chain |
| `camera.py` | Per-camera stream management, tracking, drawing |
| `routes.py` | Flask REST API + MJPEG streaming (for dashboard mode) |
| `main.py` | Entry point for full server mode |

## Quick Start — Accuracy Testing

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get your model ready

**If you have a .pt file (Ultralytics YOLO):**
```bash
python -c "
from ultralytics import YOLO
model = YOLO('your_model.pt')
model.export(format='openvino', int8=True, imgsz=640)
"
```
Copy the exported `*_openvino_model/` folder into `master/`.

**Update `config.py`** — add your model path to `MODEL_CANDIDATES`:
```python
MODEL_CANDIDATES = [
    SCRIPT_DIR / "your_model_int8_openvino_model" / "your_model.xml",
]
```

### 3. Run accuracy test

```bash
python test_accuracy.py --source /path/to/video.mp4
```

### Controls

| Key | Action |
|---|---|
| `z` | Draw queue zone (left-click points, right-click finish) |
| `x` | Draw service zone |
| `r` | Reset zones |
| `+` / `-` | Adjust confidence threshold |
| `t` | Toggle tracking |
| `b` | Toggle bounding boxes |
| `SPACE` | Pause/resume |
| `s` | Save screenshot |
| `q` | Quit |

### 4. Run full server mode (optional)

```bash
python main.py --port 7000
```

Then use the REST API to add cameras and draw zones. See `routes.py` for API docs.

## Configuration

Edit `config.py`:

| Setting | Default | What |
|---|---|---|
| `MODEL_ARCH` | `"auto"` | `"yolo26"` (NMS-free), `"yolov8"`, or `"auto"` |
| `INFERENCE_BACKEND` | `"openvino"` | `"openvino"`, `"axelera"`, or `"auto"` |
| `CONF_THR` | `0.30` | Detection confidence threshold |
| `TRACKER_BBOX_INFLATE` | `2.0` | Bounding box doubling (1.0 = off) |
| `INF_SIZE` | `640` | Inference resolution |

## How Wait Time / Processing Time Works

**With tracking (primary):** Each person gets a track ID. When they enter a zone, we record the timestamp. When they leave, we compute `dwell_time = exit_time - entry_time`. Wait time = dwell in queue zone. Processing time = dwell in service zone.

**Without tracking (fallback):** Little's Law: `W = L / λ` where L = average queue occupancy, λ = exit rate per minute.
