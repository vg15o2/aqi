"""
VideoDecoder — hardware-accelerated video decode with multi-level fallback.

Priority: GStreamer VA-API -> GStreamer QSV -> FFmpeg VA-API -> FFmpeg QSV -> CPU
"""

import os
import threading
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from config import log
from gpu_detect import (
    VAAPI_OK, QSV_OK, RENDER_DEV,
    GST_AVAILABLE, build_gst_pipeline,
)


class VideoDecoder:
    def __init__(self):
        self.cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._thread = None
        self._run = False
        self.source = None
        self.mode = "idle"
        self.fps = 0.0
        self._fc = 0
        self._t0 = 0.0
        self.decode_method = "cpu"
        self.on_loop_restart: Optional[Callable] = None

    def start(self, source: str, mode: str = "live"):
        self.stop()
        self.source = source
        self.mode = mode
        self._run = True
        self._fc = 0
        self._t0 = time.time()
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"Decode-{str(source)[:30]}",
        )
        self._thread.start()
        log.info(f"[Decoder] {mode}: {source}")

    def stop(self):
        self._run = False
        if self._thread:
            self._thread.join(timeout=3)
        if self.cap:
            self.cap.release()
            self.cap = None
        self._frame = None
        self.mode = "idle"

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    # ── GStreamer probe ──────────────────────────────────────────────────────
    def _try_gst(self, pipeline: str, label: str):
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    log.info(f"[Decoder] {label}  ({frame.shape[1]}x{frame.shape[0]})")
                    return cap
                cap.release()
        except Exception as e:
            log.debug(f"[Decoder] {label} failed: {e}")
        return None

    # ── FFmpeg probe ─────────────────────────────────────────────────────────
    def _try_ffmpeg(self, src, options: str, label: str):
        try:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
            cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    log.info(f"[Decoder] {label}: {str(src)[:45]}")
                    return cap
                cap.release()
        except Exception as e:
            log.debug(f"[Decoder] {label} failed: {e}")
        finally:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        return None

    # ── Best-effort open ─────────────────────────────────────────────────────
    def _open_best(self, src) -> Tuple[Optional[object], str]:
        src_str = str(src)
        is_stream = (src_str.startswith("rtsp") or
                     src_str.startswith("http") or
                     src_str.startswith("rtp"))

        # 1. GStreamer GPU pipelines
        if GST_AVAILABLE:
            for pipeline, label in build_gst_pipeline(src_str, is_stream, RENDER_DEV):
                cap = self._try_gst(pipeline, label)
                if cap:
                    return cap, label.lower().replace("-", "_")

        # 2. FFmpeg + VA-API
        if VAAPI_OK:
            cap = self._try_ffmpeg(
                src,
                f"video_codec;h264_vaapi|hwaccel;vaapi|hwaccel_device;{RENDER_DEV}|"
                f"hwaccel_output_format;nv12",
                "FFmpeg-VAAPI-H264",
            )
            if cap:
                return cap, "ffmpeg_vaapi"

        # 3. FFmpeg + QSV
        if QSV_OK:
            cap = self._try_ffmpeg(
                src,
                f"video_codec;h264_qsv|hwaccel;qsv|hwaccel_device;{RENDER_DEV}",
                "FFmpeg-QSV-H264",
            )
            if cap:
                return cap, "ffmpeg_qsv"

        # 4. FFmpeg plain (RTSP streams)
        if is_stream:
            try:
                cap = cv2.VideoCapture(src_str, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if cap.isOpened():
                    log.info(f"[Decoder] FFmpeg CPU (stream): {src_str[:45]}")
                    return cap, "cpu_ffmpeg"
                cap.release()
            except Exception:
                pass

        # 5. Plain CPU fallback
        try:
            cap = cv2.VideoCapture(src_str)
            if cap.isOpened():
                log.warning(f"[Decoder] CPU software decode: {src_str[:45]}")
                return cap, "cpu"
            cap.release()
        except Exception:
            pass

        return None, "none"

    # ── Main decode loop ─────────────────────────────────────────────────────
    def _loop(self):
        src = self.source
        try:
            src = int(src)
        except (ValueError, TypeError):
            pass

        def _is_local_file(s):
            if isinstance(s, int):
                return False
            return not (str(s).startswith("rtsp") or
                        str(s).startswith("http") or
                        str(s).startswith("rtp"))

        if _is_local_file(src) and self.mode != "recording":
            log.info("[Decoder] Local file detected — switching to recording mode")
            self.mode = "recording"

        def _open():
            cap, method = self._open_best(src)
            if cap is None:
                log.error(f"[Decoder] Cannot open: {self.source}")
            else:
                self.decode_method = method
                log.info(f"[Decoder] Active decode: {method}")
            return cap

        cap = _open()
        if cap is None:
            self._run = False
            return
        self.cap = cap

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps <= 0 or native_fps > 120:
            native_fps = 25.0
        frame_interval = 1.0 / native_fps
        last_read = 0.0
        loop_count = 0

        while self._run:
            now = time.time()
            if self.mode == "recording":
                elapsed = now - last_read
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

            ret, frame = cap.read()
            last_read = time.time()

            if not ret:
                if self.mode == "recording":
                    loop_count += 1
                    log.info(f"[Decoder] '{str(self.source)[:40]}' end of file — loop #{loop_count}")
                    cap.release()
                    time.sleep(0.15)
                    cap = _open()
                    if cap is None:
                        log.error("[Decoder] Cannot reopen — stopping")
                        self._run = False
                        break
                    self.cap = cap
                    native_fps = cap.get(cv2.CAP_PROP_FPS)
                    if native_fps <= 0 or native_fps > 120:
                        native_fps = 25.0
                    frame_interval = 1.0 / native_fps
                    if self.on_loop_restart:
                        try:
                            self.on_loop_restart(loop_count)
                        except Exception as e:
                            log.warning(f"[Decoder] on_loop_restart error: {e}")
                    continue
                else:
                    time.sleep(0.05)
                    continue

            with self._lock:
                self._frame = frame
            self._fc += 1
            e = time.time() - self._t0
            self.fps = self._fc / e if e > 0 else 0.0

        cap.release()
        self.cap = None
