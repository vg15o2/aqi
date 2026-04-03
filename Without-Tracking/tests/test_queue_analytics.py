"""Unit tests for QueueAnalytics predictive wait/proc time."""
import sys
import time
import importlib.util
from collections import deque
from pathlib import Path
from unittest.mock import patch, MagicMock

# Load the module without triggering real subprocess calls (vainfo / ffmpeg).
# The module runs _detect_gpu_decode() at import time, which calls subprocess.run.
_HERE = Path(__file__).parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "flow_line_single_zone",
    _HERE / "flow-line-single-zone.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
_FAKE_RUN = MagicMock(return_value=MagicMock(stdout="", stderr="", returncode=1))
with patch("subprocess.run", _FAKE_RUN):
    _SPEC.loader.exec_module(_MOD)

QueueAnalytics = _MOD.QueueAnalytics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_qa():
    """Return a QueueAnalytics instance with both zones set."""
    qa = QueueAnalytics(alert_thr=10, wait_thr=300.0, cam_id=1)
    qa.set_zones(
        [[0, 0], [0.5, 0], [0.5, 1], [0, 1]],   # queue zone (left half)
        [[0.5, 0], [1, 0], [1, 1], [0.5, 1]],   # service zone (right half)
        [],
    )
    return qa


# ---------------------------------------------------------------------------
# _recent_rate tests
# ---------------------------------------------------------------------------

def test_recent_rate_empty_deque_returns_zero():
    qa = _make_qa()
    assert qa._recent_rate(deque(), time.time()) == 0.0


def test_recent_rate_all_within_window():
    qa = _make_qa()
    now = time.time()
    times = deque([now - 60, now - 30, now - 10])
    rate = qa._recent_rate(times, now)
    # 3 exits / (90 / 60) minutes = 2.0 per minute
    assert abs(rate - 2.0) < 0.01
    assert len(times) == 3  # nothing pruned


def test_recent_rate_prunes_old_entries():
    qa = _make_qa()
    now = time.time()
    # Two entries outside the 90-second window, two inside
    times = deque([now - 200, now - 150, now - 30, now - 10])
    rate = qa._recent_rate(times, now)
    assert abs(rate - (2 / 1.5)) < 0.01
    assert len(times) == 2  # two old entries pruned


def test_recent_rate_all_outside_window_returns_zero():
    qa = _make_qa()
    now = time.time()
    times = deque([now - 200, now - 150])
    rate = qa._recent_rate(times, now)
    assert rate == 0.0
    assert len(times) == 0


# ---------------------------------------------------------------------------
# Prediction output fields in update()
# ---------------------------------------------------------------------------

def test_update_returns_prediction_fields():
    qa = _make_qa()
    result = qa.update([], 640, 480)
    assert "predicted_wait_s" in result
    assert "pred_wait_method" in result
    assert "predicted_proc_s" in result
    assert "pred_proc_method" in result


def test_prediction_falls_back_when_no_recent_exits():
    qa = _make_qa()
    result = qa.update([], 640, 480)
    assert result["pred_wait_method"] == "fallback"
    assert result["pred_proc_method"] == "fallback"


def test_prediction_uses_predictive_mode_with_sufficient_exits():
    qa = _make_qa()
    now = time.time()
    # Inject 3 recent queue exits directly (simulates debounce confirming exits)
    for _ in range(3):
        qa._queue_exit_times.append(now - 10)
    result = qa.update([], 640, 480)
    assert result["pred_wait_method"] == "predictive"


def test_prediction_proc_uses_predictive_mode_with_sufficient_service_exits():
    qa = _make_qa()
    now = time.time()
    for _ in range(3):
        qa._service_exit_times.append(now - 10)
    result = qa.update([], 640, 480)
    assert result["pred_proc_method"] == "predictive"


def test_reset_clears_exit_time_deques():
    qa = _make_qa()
    now = time.time()
    qa._queue_exit_times.append(now - 5)
    qa._service_exit_times.append(now - 5)
    qa.reset()
    assert len(qa._queue_exit_times) == 0
    assert len(qa._service_exit_times) == 0
