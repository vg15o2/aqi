"""
Airport Queue Management System — Entry Point
==============================================
Hardware pipeline (Intel Core Ultra 7 255H):
  NPU  -> YOLOv8 INT8 inference (OpenVINO worker pool)
  iGPU -> Video decode via QSV / VA-API
  CPU  -> Flask REST API, analytics, drawing

Run:
  python main.py [--host 0.0.0.0] [--port 7000]

Add streams:
  POST /api/cameras  {"name":"Gate-1","source":"rtsp://...","mode":"live","inf_fps":15}
  POST /api/cameras  {"name":"Counter-A","source":"/path/video.mp4","mode":"recording"}
"""

import argparse

from config import ALERTS_DIR, log
from gpu_detect import VAAPI_OK, GST_VAAPI_OK, GST_QSV_OK, RENDER_DEV
from inference import InferenceEngine
from camera import CameraManager
from routes import create_app


def main():
    parser = argparse.ArgumentParser(description="Airport Queue Management (Multi-Stream)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=7000, type=int)
    parser.add_argument("--alert-threshold", default=10, type=int)
    parser.add_argument("--wait-threshold", default=90.0, type=float)
    args = parser.parse_args()

    npu_engine = InferenceEngine()
    npu_engine.start()

    manager = CameraManager(npu_engine)
    app = create_app(npu_engine, manager)

    log.info(f"\n  Dashboard -> http://localhost:{args.port}")
    log.info(f"    NPU/GPU/CPU Inference via OpenVINO AUTO")
    log.info(f"    GPU Video Decode: GST-VAAPI={GST_VAAPI_OK}  GST-QSV={GST_QSV_OK}  "
             f"FFmpeg-VAAPI={VAAPI_OK}  device={RENDER_DEV}")
    log.info(f"    Zone-change debounce analytics (no tracking)")
    log.info(f"    Alerts persisted -> {ALERTS_DIR}")
    log.info(f"    APIs:")
    log.info(f"      GET  /api/overview")
    log.info(f"      GET  /api/cameras/<id>/history/export")
    log.info(f"      GET  /api/cameras/<id>/performance")
    log.info(f"      POST /api/cameras/<id>/alerts/<idx>/ack")
    log.info(f"    Add: POST /api/cameras {{name, source, mode, inf_fps}}\n")

    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
