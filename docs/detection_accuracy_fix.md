# Airport Queue Intelligence: Model Retraining & Analytics Fix Plan

## Context

The system detects people in airport queues via security cameras and estimates wait/processing times. Current problems:

1. **YOLOv8n misses people** in top-down views and produces **false positives** in angled views (conf=0.12 is too low, inflating FPs)
2. **No object tracking** — frame-by-frame zone counting with a 3-frame debounce causes count fluctuations and corrupts exit rate measurements
3. **Little's Law analytics are unreliable** — requires steady-state conditions that airport queues don't have; noisy detection makes it worse

**Edge hardware:** Intel Core Ultra 7 255H, NPU 4.0 (~48 TOPS), Arc 140T iGPU, 32GB DDR5. Target: 30-40 RTSP streams at 5 FPS inference.

**Available resources:** GPU workstation for training, edge device for deployment, 2 test videos (actual RTSP recordings).

---

## Phase A: Detection Accuracy

### A.1: Dataset Preparation (GPU workstation)

Create `training/prepare_dataset.py` that merges these into unified YOLO format (single class: person):

| Dataset | Purpose | Est. Images |
|---------|---------|-------------|
| **CrowdHuman** | Dense crowds, heavy occlusion | ~15K |
| **VisDrone-DET** | Overhead/aerial perspective (person + pedestrian classes only) | ~6K |
| **MOT17/MOT20** | Surveillance camera angles | ~10K |
| **Own camera frames** | Domain-specific fine-tuning | 200-500 |

**Own camera frames workflow:**
1. Extract every 30th frame from the 2 test videos
2. Pseudo-label with a large pretrained model (YOLOv8x)
3. Manually correct in CVAT or Label Studio
4. Export to YOLO format

**Conversion details:**
- CrowdHuman: `.odgt` annotations → YOLO format. Filter out "mask" and "ignore" regions. Map "fbox" (full body box) to class 0.
- VisDrone: Keep only class "pedestrian" (1) and "people" (2). Remap both to class 0.
- MOT17/MOT20: Ground-truth `gt.txt` format `(frame, id, x, y, w, h, conf, class, vis)` → YOLO format. Keep entries with class 1 (pedestrian) and visibility > 0.3.

Create `training/dataset.yaml`:
```yaml
path: ./datasets/airport_person
train: images/train
val: images/val
nc: 1
names: ['person']
```

90/10 train/val split, stratified by source. Expected total: ~30-50K images.

**Validation:** Verify label format with `ultralytics val` dry-run. Spot-check 50 random images per source.

---

### A.2: Model Training (GPU workstation)

Create `training/train.py`:

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # or yolo11s.pt
model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,           # adjust to GPU VRAM
    patience=15,        # early stopping
    augment=True,
    mosaic=1.0,
    mixup=0.15,
    perspective=0.001,  # heavy perspective augmentation
    degrees=10,
    scale=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    single_cls=True,    # single class (person)
    device=0,
)
```

**Key decisions:**
- **YOLOv8s vs YOLOv11s**: Train both, compare mAP on held-out test set. YOLOv11s has C2PSA attention that helps with small/occluded objects. Pick whichever scores higher.
- **Single class only**: No multi-class for v0. The backbone supports adding classes later (Phase 2: bags, children, staff) by fine-tuning only the detection head.
- **Perspective augmentation**: Critical — simulates the 30-60 degree top-down views that COCO-pretrained models struggle with.

**Targets:**
- mAP@50 > 0.75 on held-out test frames
- Recall > 0.85 at conf=0.25
- Visual inspection: detections on both top-down and angled test videos should show clear improvement

---

### A.3: INT8 Quantization & Deployment

Create `training/quantize_openvino.py`:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
model.export(
    format="openvino",
    int8=True,
    data="dataset.yaml",   # calibration data
    imgsz=640,
    half=False,
)
```

This uses Ultralytics' built-in NNCF post-training quantization. Output: `best_int8_openvino_model/` directory with `best.xml` + `best.bin`.

**Alternative (finer control):** Use OpenVINO NNCF directly:
1. Export to ONNX: `model.export(format="onnx")`
2. Convert: `mo --input_model best.onnx --compress_to_fp16`
3. Quantize: NNCF `quantize()` with 300 calibration images

**Deployment:**
1. Copy `best_int8_openvino_model/` to edge device at `Without-Tracking/best_int8_openvino_model/`
2. The existing `OV_MODEL_CANDIDATES` list (line 76-80 in `flow-line-single-zone.py`) already searches this path first
3. Run `benchmark_app -m best.xml -d NPU -nstreams 8 -hint throughput` on edge to verify NPU compatibility

**Validation:**
- Accuracy drop from FP32 should be < 1% mAP@50
- NPU inference latency: target < 5ms per frame for YOLOv8s, < 3ms for YOLOv8n
- **Critical gate:** If YOLOv8s INT8 > 5ms/inference on NPU, max aggregate throughput is 200 FPS (barely covers 40 streams x 5 FPS). In this case, **fall back to retrained YOLOv8n** — the dataset quality is the primary accuracy lever, not the architecture.

---

### A.4: Per-Camera Confidence & BBox Filters

**Why:** Different camera angles need different detection thresholds. A top-down camera needs lower confidence to catch partially visible people, while an angled camera can use higher confidence. BBox size filters reject obvious false positives (bags, reflections).

**File:** `Without-Tracking/flow-line-single-zone.py`

**Changes:**

1. **Remove global `CONF_THR = 0.12` (line 83).** Raise default to 0.25.

2. **Add per-camera config to `CameraStream.__init__` (line ~1139):**
   ```python
   self.conf_thr = 0.25           # confidence threshold (was global 0.12)
   self.min_bbox_area = 0.0005    # normalized min area (filters tiny FPs)
   self.max_bbox_area = 0.25      # normalized max area (filters giant FPs)
   self.min_bbox_hw_ratio = 0.3   # min height/width ratio (person is taller than wide)
   self.max_bbox_hw_ratio = 5.0   # max height/width ratio
   ```

3. **Pass `conf_thr` through inference pipeline:**
   - `NPUInferenceEngine.submit()` (line ~201): add `conf_thr` to the enqueued tuple
   - `NPUInferenceEngine._worker()` (line ~233): unpack and pass to `_postprocess()`
   - `NPUInferenceEngine._postprocess()` (line ~218): use passed `conf_thr` instead of global

4. **Add bbox filtering in `CameraStream._on_inference()` (line ~1189):**
   ```python
   # After receiving raw_dets, before calling analytics.update()
   filtered = []
   for d in raw_dets:
       x1, y1, x2, y2 = d["bbox"]
       bw, bh = (x2 - x1) / w, (y2 - y1) / h
       area = bw * bh
       ratio = bh / max(bw, 1e-6)
       if (self.min_bbox_area <= area <= self.max_bbox_area and
           self.min_bbox_hw_ratio <= ratio <= self.max_bbox_hw_ratio):
           filtered.append(d)
   ```

5. **Expose via REST API — `POST /api/cameras/<id>/zones` (line ~1630):**
   ```python
   cam.conf_thr = float(d.get("conf_threshold", 0.25))
   cam.min_bbox_area = float(d.get("min_bbox_area", 0.0005))
   cam.max_bbox_area = float(d.get("max_bbox_area", 0.25))
   ```

6. **Apply same changes to `flow-line-with-multiple-zone.py`.**

**Validation:** Run both test videos with old model (YOLOv8n, conf=0.12) vs new model (YOLOv8s, conf=0.25 + bbox filters). Compare detection accuracy visually and by count stability (stddev of zone count over 60s windows).

---

## Phase B: Person Tracking (ByteTrack)

### B.1: Vendor ByteTrack

Create `Without-Tracking/bytetrack.py` (~350 lines, self-contained):

**Components:**
1. **Kalman filter** (~50 lines, numpy-only): State vector `[cx, cy, aspect_ratio, height, vx, vy, va, vh]`. Predicts next position, updates with measurement.
2. **STrack class**: Represents a single tracked object. Holds Kalman state, track ID, score, activation status, lost frame count.
3. **Hungarian matching**: Uses `scipy.optimize.linear_sum_assignment` for optimal assignment. Cost matrix based on IoU distance.
4. **BYTETracker class**: Main tracker. Two-stage association:
   - Stage 1: Match high-confidence detections (score > `track_thresh`) to existing tracks via IoU
   - Stage 2: Match remaining low-confidence detections to unmatched tracks
   - Unmatched detections become new tracks
   - Unmatched tracks become "lost" — removed after `track_buffer` frames

**Config:**
```python
BYTETracker(
    track_thresh=0.25,   # high confidence threshold for first association
    track_buffer=30,     # frames to keep lost tracks (30 frames = 6s at 5fps)
    match_thresh=0.8,    # IoU threshold for matching
    frame_rate=5,        # target inference FPS (affects Kalman prediction step)
)
```

**Dependencies:** numpy (already present), scipy (for `linear_sum_assignment`).

**Why vendor instead of pip install:** Avoids external dependency issues on the edge device. The implementation is small and stable. No updates needed.

---

### B.2: Integrate into CameraStream

**File:** `Without-Tracking/flow-line-single-zone.py`

**In `CameraStream.__init__` (line ~1139):**
```python
from bytetrack import BYTETracker
self.tracker = BYTETracker(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=5)
```

**In `CameraStream._on_inference()` (line ~1189):**
```python
def _on_inference(self, cam_id, frame, raw_dets, inf_ms):
    h, w = frame.shape[:2]

    # Phase A.4: bbox size filtering
    filtered_dets = self._filter_dets(raw_dets, w, h)

    # Phase B: ByteTrack
    if filtered_dets:
        det_array = np.array([[*d["bbox"], d["conf"]] for d in filtered_dets])
        tracks = self.tracker.update(det_array, (h, w), (h, w))
        tracked_dets = [
            {"bbox": list(t.tlbr), "conf": t.score, "cls": 0, "track_id": t.track_id}
            for t in tracks if t.is_activated
        ]
    else:
        tracked_dets = []

    # Pass tracked detections to analytics
    if self.analytics.zones_defined():
        metrics = self.analytics.update(tracked_dets, w, h)
    ...
```

---

### B.3: Update QueueAnalytics for Track IDs

**File:** `Without-Tracking/flow-line-single-zone.py`, `QueueAnalytics` class

**Add to `__init__` (line ~340):**
```python
self._zone_tracks: Dict[int, str] = {}        # track_id -> "queue" or "service"
self._zone_entry_times: Dict[int, float] = {}  # track_id -> entry timestamp
self._track_last_seen: Dict[int, int] = {}     # track_id -> last frame number
self._frame_count: int = 0
```

**In `update()` (line ~474):**
- Auto-detect tracking mode: `tracking_enabled = any("track_id" in d for d in dets)`
- If tracking enabled:
  - For each detection, determine zone via `_in_zone()`
  - Record zone assignment: `_zone_tracks[track_id] = zone`
  - Record entry time if new to zone: `_zone_entry_times[track_id] = now`
  - Detect zone transitions (queue → service, service → exit)
  - Clean up stale tracks (not seen for `track_buffer` frames)
- **Run both systems in parallel during validation:** debounce counts drive displayed metrics, track-based counts are logged for comparison

**Why parallel:** Allows comparing tracker stability vs debounce stability on real data before switching over. Lower risk.

---

### B.4: Update Drawing

**File:** `Without-Tracking/flow-line-single-zone.py`, `CameraStream._draw()` (line ~1351)

- Display track ID as small text next to "Q" or "S" label on each bounding box
- Color bounding boxes by track_id (hash to hue, consistent across frames)
- This makes it visually obvious whether tracks are stable

**Validation:**
- Visual: tracks persist as people move, IDs are stable, no rapid ID switching
- Quantitative: compare stddev of zone count over 60s windows (debounce vs tracker)
- Performance: `tracker.update()` should take < 2ms/frame at 20 detections
- Memory: ~256KB total for 40 cameras (negligible)

---

## Phase C: Analytics Rewrite (Dwell-Time)

### C.1: Add Dwell-Time State

**File:** `Without-Tracking/flow-line-single-zone.py`, `QueueAnalytics.__init__` (line ~340)

```python
# Track-based dwell time
self._queue_entry_times: Dict[int, float] = {}       # track_id -> timestamp
self._service_entry_times: Dict[int, float] = {}     # track_id -> timestamp
self._queue_dwell_history: deque = deque(maxlen=200)  # completed queue dwell times (seconds)
self._service_dwell_history: deque = deque(maxlen=200)# completed service dwell times (seconds)
self._active_zone: Dict[int, str] = {}               # track_id -> current zone

# Exit time tracking (for _recent_rate, tested in test_queue_analytics.py)
self._queue_exit_times: deque = deque(maxlen=300)
self._service_exit_times: deque = deque(maxlen=300)
```

---

### C.2: Implement `_recent_rate()`

This method is already tested in `test_queue_analytics.py` (lines 43-74). Implementation:

```python
def _recent_rate(self, times: deque, now: float, window: float = 90.0) -> float:
    """Rate of events per minute within a rolling window."""
    while times and now - times[0] > window:
        times.popleft()
    if not times:
        return 0.0
    span = now - times[0]
    return len(times) / (span / 60.0) if span > 0 else 0.0
```

---

### C.3: Rewrite `update()` Core Logic

Replace Little's Law (lines ~587-651) with track-based dwell measurement.

**New flow:**

```
STEP 1: For each tracked detection, determine zone via _in_zone()

STEP 2: Zone transition logic per track_id:
  - NEW in queue zone:
      queue_entry_times[track_id] = now
      active_zone[track_id] = "queue"
  - MOVED queue -> service:
      queue_dwell = now - queue_entry_times.pop(track_id)
      queue_dwell_history.append(queue_dwell)
      queue_exit_times.append(now)
      service_entry_times[track_id] = now
      active_zone[track_id] = "service"
  - MOVED service -> outside:
      service_dwell = now - service_entry_times.pop(track_id)
      service_dwell_history.append(service_dwell)
      service_exit_times.append(now)
      del active_zone[track_id]
  - DISAPPEARED (not seen for track_buffer frames):
      treat as exit from current zone, record dwell

STEP 3: Compute metrics:
  queue_length = count of tracks currently in queue zone

  # Current wait times for people still in queue
  current_waits = [now - entry for entry in queue_entry_times.values()]
  avg_waiting_time = mean(current_waits) if current_waits else 0

  # Historical dwell times (completed waits)
  wait_p50 = percentile(queue_dwell_history, 50)
  wait_p90 = percentile(queue_dwell_history, 90)

  # Predicted wait for someone joining now
  queue_exit_rate = _recent_rate(queue_exit_times, now)
  if queue_exit_rate > 0:
      predicted_wait = queue_length / queue_exit_rate * 60
      pred_method = "predictive"
  else:
      predicted_wait = mean(queue_dwell_history) if queue_dwell_history else 0
      pred_method = "fallback"

  # Processing time
  avg_processing_time = mean(service_dwell_history) if service_dwell_history else 0

  # Throughput
  service_exit_rate = _recent_rate(service_exit_times, now)
  throughput_per_hour = service_exit_rate * 60
```

**Graceful degradation:** If detections lack `track_id` (no tracker available), fall back to the existing debounce-based counting. Auto-detect: `tracking_enabled = any("track_id" in d for d in dets)`.

---

### C.4: New Return Fields

Add to `update()` return dict (line ~688):

```python
"predicted_wait_s":     predicted_wait,
"pred_wait_method":     pred_method,        # "predictive" or "fallback"
"predicted_proc_s":     predicted_proc,
"pred_proc_method":     pred_proc_method,
"wait_p50_s":           wait_p50,
"wait_p90_s":           wait_p90,
"proc_p50_s":           proc_p50,
"proc_p90_s":           proc_p90,
"active_queue_waits":   {tid: round(now - t, 1) for tid, t in queue_entry_times.items()},
"active_service_waits": {tid: round(now - t, 1) for tid, t in service_entry_times.items()},
```

---

### C.5: Update `reset()` / `smooth_reset()`

**`reset()` (line ~782):** Clear all new dicts and deques.

**`smooth_reset()` (line ~792):** Preserve `_queue_dwell_history` and `_service_dwell_history` (completed dwells remain valid). Flush active entry times only if tracks are truly lost.

---

### C.6: Make Existing Tests Pass

Tests in `Without-Tracking/tests/test_queue_analytics.py` define the contract:
- `_recent_rate()` — empty deque, within-window, pruning, outside-window (lines 43-74)
- `update()` returns `predicted_wait_s`, `pred_wait_method`, `predicted_proc_s`, `pred_proc_method` (lines 81-87)
- Prediction mode switches to "predictive" when 3+ recent exits detected (lines 90-113)
- `reset()` clears `_queue_exit_times` and `_service_exit_times` (lines 116-123)

Run: `pytest Without-Tracking/tests/test_queue_analytics.py -v`

---

### C.7: Update Dashboard

**File:** `Without-Tracking/flow-line-single-zone.html`

- Add P50 and P90 wait time display in the metrics panel
- Show per-person active wait times (small table: track_id, current wait)
- Switch main "Avg Wait" display to use `predicted_wait_s` when available, fallback to `avg_waiting_time`

---

## Phase D: Throughput Validation

### D.1: NPU Baseline (run early, after A.3)

```bash
benchmark_app -m best_int8_openvino_model/best.xml -d NPU -nstreams 8 -hint throughput -niter 1000 -b 1
```

Compare YOLOv8n vs YOLOv8s latency on NPU. **If YOLOv8s > 5ms/inference**, fall back to retrained YOLOv8n.

---

### D.2: Decode Benchmark

Create `Without-Tracking/decode_bench.py`:
- Open N concurrent `VideoDecoder` instances reading the same test video
- Measure aggregate decode FPS per backend (QSV, VA-API, CPU)
- Test at N = 10, 20, 30, 40
- Find the max streams the Arc 140T can decode at >= 5 FPS each

---

### D.3: Full Load Test

Create `Without-Tracking/loadtest.py`:
- Start `NPUInferenceEngine` + N `CameraStream` instances replaying test videos
- Run for 60 seconds at each concurrency level (10, 20, 30, 40)
- Measure:

| Metric | Target |
|--------|--------|
| Per-stream inference FPS | >= 3 FPS |
| P95 inference latency | < 200ms |
| Memory RSS | < 24GB |
| Flask API response time | < 100ms |
| CPU utilization | < 80% |

---

### D.4: Tuning

Based on load test results, adjust:
- `NPUInferenceEngine.NUM_WORKERS` (default 8)
- `frame_queue maxsize` (default 128)
- OpenVINO `PERFORMANCE_HINT` and `NUM_STREAMS`
- Per-stream `inf_fps_limit` (may need to drop from 5 to 3 if NPU saturates)
- Decode resolution (drop to 720p if memory bandwidth is bottleneck)

---

## Sequencing

```
Week 1-2: A.1 + A.2  (dataset prep + model training on GPU workstation)
Week 2:   A.3         (INT8 quantize + deploy to edge)
Week 3:   A.4 + D.1   (per-camera config + NPU baseline benchmark)
Week 4:   B.1-B.4     (ByteTrack integration)
Week 5:   C.1-C.7     (analytics rewrite + dashboard updates)
Week 6:   D.2-D.4     (full load test + tuning)
```

D.1 runs early (week 3) to catch model-too-large issues before investing in B/C.

---

## Files Modified

| File | Phases |
|------|--------|
| `Without-Tracking/flow-line-single-zone.py` | A.4, B, C |
| `Without-Tracking/flow-line-single-zone.html` | C.7 |
| `Without-Tracking/flow-line-with-multiple-zone.py` | A.4, B, C (parallel port) |

## New Files

| File | Phase | Description |
|------|-------|-------------|
| `training/prepare_dataset.py` | A.1 | Merges CrowdHuman + VisDrone + MOT into YOLO format |
| `training/train.py` | A.2 | Ultralytics training script |
| `training/quantize_openvino.py` | A.3 | Export + INT8 quantization |
| `training/dataset.yaml` | A.1 | Dataset config |
| `Without-Tracking/bytetrack.py` | B.1 | Vendored ByteTrack (~350 lines) |
| `Without-Tracking/loadtest.py` | D.3 | Multi-stream load test |
| `Without-Tracking/decode_bench.py` | D.2 | Decode capacity benchmark |

## Fallback Strategy

If YOLOv8s is too heavy for 40 streams on NPU: **retrain YOLOv8n on the improved dataset**. The dataset quality (CrowdHuman + VisDrone + overhead angles) is the primary accuracy lever — architecture is secondary. A retrained YOLOv8n with the right data + per-camera confidence tuning will still be a major improvement over the current model.

## Phase 2 Extensibility

The YOLOv8s/v11s backbone supports adding multi-class detection heads later (bags, children, staff) by fine-tuning only the head layers. The ByteTrack tracker and analytics engine are class-agnostic — they accept a `cls` field and can be extended to track multiple classes. No architectural changes needed now to support Phase 2.
