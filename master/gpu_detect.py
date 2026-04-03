"""
Intel iGPU decode capability detection and GStreamer pipeline builder.

Probes for VA-API, QSV, and GStreamer plugins at startup so that
VideoDecoder can pick the best hardware decode path.
"""

import subprocess
from pathlib import Path
from typing import Tuple

import cv2

from config import log


# ─────────────────────────────────────────────────────────────────────────────
# VA-API / QSV detection (run once at import)
# ─────────────────────────────────────────────────────────────────────────────
def _detect_gpu_decode() -> Tuple[bool, bool, str]:
    render_device = "/dev/dri/renderD128"
    try:
        dri = Path("/dev/dri")
        if dri.exists():
            render_nodes = sorted(dri.glob("renderD*"))
            if render_nodes:
                render_device = str(render_nodes[0])
    except Exception:
        pass

    vaapi_ok = False
    try:
        r = subprocess.run(
            ["vainfo", "--display", "drm", "--device", render_device],
            capture_output=True, text=True, timeout=5,
        )
        vaapi_ok = "VAEntrypointVLD" in r.stdout
        if vaapi_ok:
            log.info(f"[GPU] VA-API available on {render_device}")
    except Exception as e:
        log.info(f"[GPU] VA-API check failed: {e}")

    qsv_ok = False
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=5,
        )
        qsv_ok = "qsv" in (r.stdout + r.stderr).lower()
        if qsv_ok:
            log.info("[GPU] QSV available via FFmpeg")
    except Exception as e:
        log.info(f"[GPU] QSV check failed: {e}")

    return vaapi_ok, qsv_ok, render_device


VAAPI_OK, QSV_OK, RENDER_DEV = _detect_gpu_decode()
log.info(f"[GPU] Decode backend: QSV={QSV_OK}  VA-API={VAAPI_OK}  device={RENDER_DEV}")


# ─────────────────────────────────────────────────────────────────────────────
# GStreamer plugin detection
# ─────────────────────────────────────────────────────────────────────────────
def _check_gstreamer() -> bool:
    try:
        build = cv2.getBuildInformation()
        return "GStreamer" in build and "YES" in build[build.find("GStreamer"):build.find("GStreamer") + 40]
    except Exception:
        return False


def _check_vaapi_gst() -> bool:
    try:
        r = subprocess.run(["gst-inspect-1.0", "vaapidecodebin"],
                           capture_output=True, timeout=4)
        if r.returncode == 0:
            return True
        r = subprocess.run(["gst-inspect-1.0", "vah264dec"],
                           capture_output=True, timeout=4)
        return r.returncode == 0
    except Exception:
        return False


def _check_qsv_gst() -> bool:
    try:
        r = subprocess.run(["gst-inspect-1.0", "msdkh264dec"],
                           capture_output=True, timeout=4)
        return r.returncode == 0
    except Exception:
        return False


GST_AVAILABLE = _check_gstreamer()
GST_VAAPI_OK  = _check_vaapi_gst() if GST_AVAILABLE else False
GST_QSV_OK    = _check_qsv_gst()   if GST_AVAILABLE else False

log.info(f"[GPU] GStreamer: {GST_AVAILABLE}  GST-VA-API: {GST_VAAPI_OK}  GST-QSV: {GST_QSV_OK}")


# ─────────────────────────────────────────────────────────────────────────────
# GStreamer pipeline builder
# ─────────────────────────────────────────────────────────────────────────────
def build_gst_pipeline(src: str, is_stream: bool, render_dev: str) -> list:
    """
    Build GStreamer pipeline strings ordered by reliability for Intel iGPU.
    Returns list of (pipeline_string, label) tuples to try in order.
    """
    pipelines = []

    if is_stream:
        src_h264 = (f'rtspsrc location="{src}" latency=50 '
                    f'! rtph264depay ! h264parse')
        src_h265 = (f'rtspsrc location="{src}" latency=50 '
                    f'! rtph265depay ! h265parse')
    else:
        src_h264 = f'filesrc location="{src}" ! qtdemux name=d d.video_0 ! h264parse'
        src_h265 = f'filesrc location="{src}" ! qtdemux name=d d.video_0 ! h265parse'

    if GST_VAAPI_OK:
        pipelines.append((
            f'{src_h264} ! vaapidecodebin ! vaapipostproc ! '
            f'videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0',
            "GST-VAAPI-H264"))

    if GST_VAAPI_OK and not is_stream:
        pipelines.append((
            f'{src_h265} ! vaapidecodebin ! vaapipostproc ! '
            f'videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0',
            "GST-VAAPI-H265"))

    if GST_VAAPI_OK:
        pipelines.append((
            f'{src_h264} ! vaapidecodebin ! '
            f'videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0',
            "GST-VAAPI-DIRECT"))

    if GST_VAAPI_OK:
        pipelines.append((
            f'{src_h264} ! vah264dec ! '
            f'videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0',
            "GST-VA-H264"))

    if GST_VAAPI_OK and not is_stream:
        pipelines.append((
            f'filesrc location="{src}" ! decodebin ! '
            f'vaapipostproc ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0',
            "GST-VAAPI-DECODEBIN"))

    if GST_QSV_OK:
        pipelines.append((
            f'{src_h264} ! msdkh264dec ! '
            f'videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0',
            "GST-QSV-H264"))

    if GST_AVAILABLE and not is_stream:
        pipelines.append((
            f'filesrc location="{src}" ! decodebin ! '
            f'videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0',
            "GST-AUTO"))

    return pipelines
