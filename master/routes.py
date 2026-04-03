"""
Flask REST API + MJPEG streaming endpoints.
"""

import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory, abort

from config import SCRIPT_DIR, log
from gpu_detect import VAAPI_OK, QSV_OK, GST_VAAPI_OK, GST_QSV_OK, GST_AVAILABLE, RENDER_DEV
from inference import InferenceEngine
from camera import CameraManager


def create_app(npu_engine: InferenceEngine, manager: CameraManager) -> Flask:
    app = Flask(__name__, static_folder=str(SCRIPT_DIR))

    def _placeholder_jpg(msg="No video -- add a camera stream"):
        ph = np.full((480, 640, 3), 30, np.uint8)
        cv2.putText(ph, msg, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        _, buf = cv2.imencode(".jpg", ph)
        return buf.tobytes()

    # ── Dashboard ────────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return send_from_directory(str(SCRIPT_DIR), "flow-lin.html")

    # ── Camera CRUD ──────────────────────────────────────────────────────────
    @app.route("/api/cameras", methods=["GET"])
    def list_cameras():
        return jsonify(manager.list())

    @app.route("/api/cameras", methods=["POST"])
    def add_camera():
        d = request.get_json(force=True)
        name = d.get("name", "Camera").strip() or "Camera"
        source = d.get("source", "").strip()
        mode = d.get("mode", "live")
        if not source:
            return jsonify({"error": "source required"}), 400
        at = int(d.get("alert_threshold", 10))
        wt = float(d.get("wait_time_threshold", 90.0))
        inf_fps = float(d.get("inf_fps", 15.0))
        cam = manager.add(name, source, mode, at, wt, inf_fps)
        return jsonify(cam.to_dict()), 201

    @app.route("/api/cameras/<int:cam_id>", methods=["DELETE"])
    def remove_camera(cam_id: int):
        manager.remove(cam_id)
        return jsonify({"status": "removed", "cam_id": cam_id})

    # ── Zones ────────────────────────────────────────────────────────────────
    @app.route("/api/cameras/<int:cam_id>/zones", methods=["POST"])
    def set_zones(cam_id: int):
        cam = manager.get(cam_id)
        if not cam:
            abort(404)
        d = request.get_json(force=True)
        cam.set_zones(
            d.get("zone_queue", []), d.get("zone_service", []), d.get("zone_exit", []),
            d.get("zone_queue_2", []), d.get("zone_service_2", []),
        )
        if d.get("counter_line") or d.get("exit_line"):
            cam.set_lines(d.get("counter_line", []), d.get("exit_line", []))
        at = int(d.get("alert_threshold", cam.analytics.alert_thr))
        wt = float(d.get("wait_time_threshold", cam.analytics.wait_thr))
        cam.set_thresholds(at, wt)
        # Per-camera detection filters
        det_filter = d.get("det_filter")
        if det_filter and isinstance(det_filter, dict):
            cam.set_det_filter(det_filter)
        # Tracking toggle
        if "tracking_enabled" in d:
            cam.tracking_enabled = bool(d["tracking_enabled"])
        cam.inference_active = True
        log.info(f"[Cam-{cam_id}] Zones applied — inference STARTED "
                 f"(tracking={'ON' if cam.tracking_enabled else 'OFF'})")
        return jsonify({"status": "ok", "cam_id": cam_id, "inference_active": True,
                        "tracking_enabled": cam.tracking_enabled, "warnings": []})

    @app.route("/api/cameras/<int:cam_id>/zones", methods=["GET"])
    def get_zones(cam_id: int):
        cam = manager.get(cam_id)
        if not cam:
            abort(404)
        a = cam.analytics
        a2 = cam.analytics_2
        return jsonify({
            "zone_queue": a.zone_queue, "zone_service": a.zone_service,
            "zone_exit": a.zone_exit,
            "alert_threshold": a.alert_thr, "wait_time_threshold": a.wait_thr,
            "zone_queue_2": a2.zone_queue, "zone_service_2": a2.zone_service,
        })

    @app.route("/api/cameras/<int:cam_id>/zones", methods=["DELETE"])
    def clear_zones(cam_id: int):
        cam = manager.get(cam_id)
        if not cam:
            abort(404)
        cam.inference_active = False
        cam.set_zones([], [], [], [], [])
        cam.analytics.reset()
        cam.analytics_2.reset()
        with cam._lock:
            cam._ann = None
            cam._metrics = {}
        log.info(f"[Cam-{cam_id}] Zones cleared — inference STOPPED")
        return jsonify({"status": "zones_cleared", "cam_id": cam_id, "inference_active": False})

    # ── Lines ────────────────────────────────────────────────────────────────
    @app.route("/api/cameras/<int:cam_id>/lines", methods=["POST"])
    def set_lines(cam_id: int):
        cam = manager.get(cam_id)
        if not cam:
            abort(404)
        d = request.get_json(force=True)
        counter_line = d.get("counter_line", [])
        exit_line = d.get("exit_line", [])
        cam.set_lines(counter_line, exit_line)
        log.info(f"[Cam-{cam_id}] Lines set")
        return jsonify({"status": "ok", "cam_id": cam_id,
                        "counter_line": counter_line, "exit_line": exit_line})

    @app.route("/api/cameras/<int:cam_id>/lines", methods=["GET"])
    def get_lines(cam_id: int):
        cam = manager.get(cam_id)
        if not cam:
            abort(404)
        a = cam.analytics
        return jsonify({
            "counter_line": a.counter_line, "exit_line": a.exit_line,
            "lines_defined": a.lines_defined(),
        })

    # ── Status / History / Alerts ────────────────────────────────────────────
    @app.route("/api/cameras/<int:cam_id>/status")
    def cam_status(cam_id: int):
        cam = manager.get(cam_id)
        if not cam:
            abort(404)
        m = cam.get_metrics()
        m.update(cam.to_dict())
        m["npu_device"] = npu_engine.actual_dev
        m["alerts"] = [{k: v for k, v in a.items() if k != "_ts"} for a in m.get("alerts", [])]
        return jsonify(m)

    @app.route("/api/cameras/<int:cam_id>/history")
    def cam_history(cam_id: int):
        cam = manager.get(cam_id)
        if not cam:
            abort(404)
        return jsonify(cam.get_history())

    @app.route("/api/cameras/<int:cam_id>/history/export")
    def cam_history_export(cam_id: int):
        cam = manager.get(cam_id)
        if not cam:
            abort(404)
        csv_data = cam.get_history_csv()
        fname = f"cam_{cam_id}_{time.strftime('%Y%m%d_%H%M%S')}_history.csv"
        return Response(csv_data, mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={fname}"})

    @app.route("/api/cameras/<int:cam_id>/alerts")
    def cam_alerts(cam_id: int):
        cam = manager.get(cam_id)
        if not cam:
            abort(404)
        return jsonify(cam.analytics.get_alert_history())

    @app.route("/api/cameras/<int:cam_id>/alerts/<int:alert_idx>/ack", methods=["POST"])
    def ack_alert(cam_id: int, alert_idx: int):
        cam = manager.get(cam_id)
        if not cam:
            abort(404)
        ok = cam.analytics.acknowledge_alert(alert_idx)
        if not ok:
            return jsonify({"error": "alert not found"}), 404
        return jsonify({"status": "acknowledged", "cam_id": cam_id, "alert_idx": alert_idx})

    @app.route("/api/cameras/<int:cam_id>/performance")
    def cam_performance(cam_id: int):
        cam = manager.get(cam_id)
        if not cam:
            abort(404)
        perf = cam.analytics.get_counter_performance()
        perf["cam_id"] = cam_id
        perf["cam_name"] = cam.name
        perf["exit_zone_defined"] = cam.analytics.exit_zone_defined()
        return jsonify(perf)

    @app.route("/api/cameras/<int:cam_id>/reset", methods=["POST"])
    def cam_reset(cam_id: int):
        cam = manager.get(cam_id)
        if not cam:
            abort(404)
        cam.reset()
        return jsonify({"status": "reset"})

    # ── Overview ─────────────────────────────────────────────────────────────
    @app.route("/api/overview")
    def overview():
        with manager._lock:
            cams = list(manager.cameras.values())
        rows = []
        total_queue = 0
        total_processed = 0
        total_unacked = 0
        for cam in cams:
            m = cam.get_metrics()
            perf = cam.analytics.get_counter_performance()
            alerts_all = cam.analytics.get_alert_history()
            unacked = [a for a in alerts_all if not a.get("acked")]
            ql = m.get("queue_length", 0)
            total_queue += ql
            total_processed += m.get("total_processed", 0)
            total_unacked += len(unacked)
            rows.append({
                "cam_id": cam.cam_id, "name": cam.name, "mode": cam.mode,
                "inference_active": cam.inference_active,
                "zones_defined": cam.analytics.zones_defined(),
                "exit_zone_defined": cam.analytics.exit_zone_defined(),
                "decode_method": cam.decoder.decode_method,
                "queue_length": ql,
                "avg_waiting_time": m.get("avg_waiting_time", 0),
                "wait_method": m.get("wait_method", "none"),
                "open_counter": m.get("open_counter", False),
                "avg_processing_time": m.get("avg_processing_time", 0),
                "throughput_per_hour": m.get("throughput_per_hour", 0),
                "tput_method": m.get("tput_method", "none"),
                "total_processed": m.get("total_processed", 0),
                "zone_counts": m.get("zone_counts", {}),
                "queue_entries": m.get("queue_entries", 0),
                "queue_exits": m.get("queue_exits", 0),
                "unacked_alerts": len(unacked),
                "counter_perf": perf,
                "inf_fps": round(cam._inf_fps, 1),
                "inf_ms": round(cam._inf_ms, 1),
                "inf_fps_limit": cam.inf_fps_limit,
            })
        return jsonify({
            "cameras": rows, "total_cameras": len(rows),
            "total_queue": total_queue, "total_processed": total_processed,
            "total_unacked_alerts": total_unacked,
            "npu_device": npu_engine.actual_dev,
            "gpu_decode_vaapi": VAAPI_OK, "gpu_decode_qsv": QSV_OK,
            "render_device": RENDER_DEV,
            "timestamp": time.strftime("%H:%M:%S"),
        })

    # ── Video feed ───────────────────────────────────────────────────────────
    @app.route("/video_feed/<int:cam_id>")
    def video_feed(cam_id: int):
        def gen():
            while True:
                cam = manager.get(cam_id)
                jpg = cam.get_jpeg() if cam else None
                if jpg is None:
                    jpg = _placeholder_jpg()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(1.0 / 25)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # ── System stats ─────────────────────────────────────────────────────────
    @app.route("/api/system")
    def sys_stats():
        return jsonify({
            "npu": npu_engine.get_stats(),
            "cameras": len(manager.cameras),
            "gpu_decode": {
                "vaapi": VAAPI_OK, "qsv": QSV_OK,
                "gst_vaapi": GST_VAAPI_OK, "gst_qsv": GST_QSV_OK,
                "gst": GST_AVAILABLE, "device": RENDER_DEV,
            },
        })

    return app
