"""
ByteTrack — lightweight multi-object tracker.

Two-stage association:
  Stage 1: High-confidence detections matched to existing tracks via IoU
  Stage 2: Low-confidence detections matched to remaining tracks
  Unmatched high-conf detections start new tracks.

References:
  Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022
"""

import numpy as np
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Kalman Filter (8-state: cx, cy, aspect_ratio, height, + velocities)
# ─────────────────────────────────────────────────────────────────────────────
class KalmanFilter:
    """Constant-velocity Kalman filter in (cx, cy, a, h) space."""

    _STD_WEIGHT_POSITION = 1.0 / 20
    _STD_WEIGHT_VELOCITY = 1.0 / 160

    def __init__(self):
        dt = 1.0
        # state transition
        self._F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self._F[i, i + 4] = dt
        # measurement matrix (observe cx, cy, a, h)
        self._H = np.eye(4, 8, dtype=np.float64)

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create state from first measurement [cx, cy, a, h]."""
        mean = np.zeros(8, dtype=np.float64)
        mean[:4] = measurement
        std = [
            2 * self._STD_WEIGHT_POSITION * measurement[3],
            2 * self._STD_WEIGHT_POSITION * measurement[3],
            1e-2,
            2 * self._STD_WEIGHT_POSITION * measurement[3],
            10 * self._STD_WEIGHT_VELOCITY * measurement[3],
            10 * self._STD_WEIGHT_VELOCITY * measurement[3],
            1e-5,
            10 * self._STD_WEIGHT_VELOCITY * measurement[3],
        ]
        cov = np.diag(np.square(std))
        return mean, cov

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std = [
            self._STD_WEIGHT_POSITION * mean[3],
            self._STD_WEIGHT_POSITION * mean[3],
            1e-2,
            self._STD_WEIGHT_POSITION * mean[3],
            self._STD_WEIGHT_VELOCITY * mean[3],
            self._STD_WEIGHT_VELOCITY * mean[3],
            1e-5,
            self._STD_WEIGHT_VELOCITY * mean[3],
        ]
        Q = np.diag(np.square(std))
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + Q
        return mean, cov

    def update(self, mean: np.ndarray, cov: np.ndarray,
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std = [
            self._STD_WEIGHT_POSITION * mean[3],
            self._STD_WEIGHT_POSITION * mean[3],
            1e-2,
            self._STD_WEIGHT_POSITION * mean[3],
        ]
        R = np.diag(np.square(std))
        S = self._H @ cov @ self._H.T + R
        K = cov @ self._H.T @ np.linalg.inv(S)
        innovation = measurement - self._H @ mean
        mean = mean + K @ innovation
        cov = (np.eye(8) - K @ self._H) @ cov
        return mean, cov


# shared instance
_KF = KalmanFilter()


# ─────────────────────────────────────────────────────────────────────────────
# Single Track
# ─────────────────────────────────────────────────────────────────────────────
class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    _next_id = 1

    def __init__(self, tlbr: np.ndarray, score: float):
        self.track_id = 0  # assigned on activation
        self.tlbr = tlbr.copy()
        self.score = score
        self.state = TrackState.New
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self._mean: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None

    # ── conversions ──────────────────────────────────────────────────────────
    @staticmethod
    def tlbr_to_xyah(tlbr):
        """[x1,y1,x2,y2] -> [cx, cy, aspect_ratio, height]"""
        w = tlbr[2] - tlbr[0]
        h = tlbr[3] - tlbr[1]
        cx = tlbr[0] + w / 2
        cy = tlbr[1] + h / 2
        a = w / (h + 1e-6)
        return np.array([cx, cy, a, h], dtype=np.float64)

    @staticmethod
    def xyah_to_tlbr(xyah):
        cx, cy, a, h = xyah
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    # ── lifecycle ────────────────────────────────────────────────────────────
    def activate(self, frame_id: int):
        self.track_id = STrack._next_id
        STrack._next_id += 1
        m = self.tlbr_to_xyah(self.tlbr)
        self._mean, self._cov = _KF.initiate(m)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 0

    def re_activate(self, new_track: "STrack", frame_id: int):
        m = self.tlbr_to_xyah(new_track.tlbr)
        self._mean, self._cov = _KF.update(self._mean, self._cov, m)
        self.tlbr = new_track.tlbr.copy()
        self.score = new_track.score
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.tracklet_len = 0

    def predict(self):
        if self._mean is not None:
            self._mean, self._cov = _KF.predict(self._mean, self._cov)
            self.tlbr = self.xyah_to_tlbr(self._mean[:4])

    def update(self, new_track: "STrack", frame_id: int):
        m = self.tlbr_to_xyah(new_track.tlbr)
        self._mean, self._cov = _KF.update(self._mean, self._cov, m)
        self.tlbr = self.xyah_to_tlbr(self._mean[:4])
        self.score = new_track.score
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.tracklet_len += 1

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def reset_id():
        STrack._next_id = 1


# ─────────────────────────────────────────────────────────────────────────────
# IoU helpers
# ─────────────────────────────────────────────────────────────────────────────
def _iou_batch(atlbrs: np.ndarray, btlbrs: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of boxes. Returns (N, M) matrix."""
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    a = np.asarray(atlbrs, dtype=np.float32)
    b = np.asarray(btlbrs, dtype=np.float32)
    # intersection
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(br - tl, 0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    # union
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Linear Assignment (greedy, no scipy dependency)
# ─────────────────────────────────────────────────────────────────────────────
def _linear_assignment(cost: np.ndarray, thresh: float):
    """
    Greedy matching on a cost matrix.
    Returns: (matches, unmatched_a, unmatched_b)
    matches: list of (row, col) pairs
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    rows, cols = cost.shape
    matched_a = set()
    matched_b = set()
    matches = []

    # flatten and sort by cost ascending
    indices = np.argsort(cost.ravel())
    for idx in indices:
        r, c = divmod(int(idx), cols)
        if cost[r, c] > thresh:
            break
        if r not in matched_a and c not in matched_b:
            matches.append((r, c))
            matched_a.add(r)
            matched_b.add(c)
            if len(matched_a) == rows or len(matched_b) == cols:
                break

    unmatched_a = [i for i in range(rows) if i not in matched_a]
    unmatched_b = [j for j in range(cols) if j not in matched_b]
    return matches, unmatched_a, unmatched_b


# ─────────────────────────────────────────────────────────────────────────────
# ByteTracker
# ─────────────────────────────────────────────────────────────────────────────
class ByteTracker:
    """
    Parameters
    ----------
    track_thresh : float
        High-confidence threshold for stage-1 association.
    match_thresh : float
        IoU threshold for matching (cost = 1 - IoU; match if cost < match_thresh).
    track_buffer : int
        Number of frames to keep a lost track before removal.
    low_thresh : float
        Minimum confidence to participate in stage-2 association.
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        low_thresh: float = 0.1,
        bbox_inflate: float = 1.0,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.low_thresh = low_thresh
        self.bbox_inflate = bbox_inflate  # >1.0 enables bbox doubling

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0

    @staticmethod
    def _inflate_bbox(tlbr: np.ndarray, factor: float) -> np.ndarray:
        """Inflate bbox by factor around center. factor=2.0 doubles w and h."""
        cx = (tlbr[0] + tlbr[2]) / 2
        cy = (tlbr[1] + tlbr[3]) / 2
        hw = (tlbr[2] - tlbr[0]) / 2 * factor
        hh = (tlbr[3] - tlbr[1]) / 2 * factor
        return np.array([cx - hw, cy - hh, cx + hw, cy + hh], dtype=np.float32)

    @staticmethod
    def _deflate_bbox(inflated_tlbr: np.ndarray, factor: float) -> np.ndarray:
        """Reverse inflate — shrink back to original size."""
        return ByteTracker._inflate_bbox(inflated_tlbr, 1.0 / factor)

    def update(self, dets: List[dict]) -> List[STrack]:
        """
        Process one frame of detections.

        Parameters
        ----------
        dets : list of dict
            Each dict has 'bbox' [x1,y1,x2,y2] and 'conf' float.

        Returns
        -------
        list of STrack — active tracks this frame (state == Tracked).
        """
        self.frame_id += 1
        activated = []
        refound = []
        lost = []
        removed = []

        # ── Split detections by confidence ───────────────────────────────────
        # Store original bboxes for output, inflate for tracking IoU
        inflate = self.bbox_inflate > 1.0
        if dets:
            bboxes_orig = np.array([d["bbox"] for d in dets], dtype=np.float32)
            scores = np.array([d["conf"] for d in dets], dtype=np.float32)
            if inflate:
                bboxes = np.array([self._inflate_bbox(b, self.bbox_inflate)
                                   for b in bboxes_orig], dtype=np.float32)
            else:
                bboxes = bboxes_orig
        else:
            bboxes = np.empty((0, 4), dtype=np.float32)
            bboxes_orig = bboxes
            scores = np.empty(0, dtype=np.float32)

        high_mask = scores >= self.track_thresh
        low_mask = (~high_mask) & (scores >= self.low_thresh)

        high_dets = [STrack(bboxes[i], scores[i]) for i in np.where(high_mask)[0]]
        low_dets = [STrack(bboxes[i], scores[i]) for i in np.where(low_mask)[0]]
        # Keep mapping from inflated STrack → original bbox for output
        _orig_map = {}
        if inflate:
            for i in np.where(high_mask)[0]:
                _orig_map[id(high_dets[int(np.sum(high_mask[:i+1])) - 1])] = bboxes_orig[i]
            for i in np.where(low_mask)[0]:
                _orig_map[id(low_dets[int(np.sum(low_mask[:i+1])) - 1])] = bboxes_orig[i]

        # ── Predict existing tracks ──────────────────────────────────────────
        strack_pool = self.tracked_stracks + self.lost_stracks
        for t in strack_pool:
            t.predict()

        tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        # ── Stage 1: High-conf detections vs tracked tracks ──────────────────
        if tracked_stracks and high_dets:
            iou_mat = _iou_batch(
                np.array([t.tlbr for t in tracked_stracks]),
                np.array([d.tlbr for d in high_dets]),
            )
            cost = 1.0 - iou_mat
            matches_1, u_tracks_1, u_dets_1 = _linear_assignment(cost, self.match_thresh)
        else:
            matches_1 = []
            u_tracks_1 = list(range(len(tracked_stracks)))
            u_dets_1 = list(range(len(high_dets)))

        for ti, di in matches_1:
            tracked_stracks[ti].update(high_dets[di], self.frame_id)
            activated.append(tracked_stracks[ti])

        # ── Stage 2: Low-conf detections vs remaining tracked tracks ─────────
        remaining_tracks = [tracked_stracks[i] for i in u_tracks_1]
        if remaining_tracks and low_dets:
            iou_mat = _iou_batch(
                np.array([t.tlbr for t in remaining_tracks]),
                np.array([d.tlbr for d in low_dets]),
            )
            cost = 1.0 - iou_mat
            matches_2, u_tracks_2, _ = _linear_assignment(cost, self.match_thresh)
        else:
            matches_2 = []
            u_tracks_2 = list(range(len(remaining_tracks)))

        for ti, di in matches_2:
            remaining_tracks[ti].update(low_dets[di], self.frame_id)
            activated.append(remaining_tracks[ti])

        # Mark still-unmatched tracked tracks as lost
        for i in u_tracks_2:
            t = remaining_tracks[i]
            if t.state != TrackState.Lost:
                t.mark_lost()
                lost.append(t)

        # ── Stage 3: Unmatched high-conf dets vs lost tracks ─────────────────
        unmatched_high = [high_dets[i] for i in u_dets_1]
        if self.lost_stracks and unmatched_high:
            iou_mat = _iou_batch(
                np.array([t.tlbr for t in self.lost_stracks]),
                np.array([d.tlbr for d in unmatched_high]),
            )
            cost = 1.0 - iou_mat
            matches_3, u_lost, u_dets_3 = _linear_assignment(cost, self.match_thresh)
        else:
            matches_3 = []
            u_lost = list(range(len(self.lost_stracks)))
            u_dets_3 = list(range(len(unmatched_high)))

        for ti, di in matches_3:
            self.lost_stracks[ti].re_activate(unmatched_high[di], self.frame_id)
            refound.append(self.lost_stracks[ti])

        # ── New tracks from remaining unmatched high-conf detections ─────────
        for i in u_dets_3:
            det = unmatched_high[i]
            if det.score >= self.track_thresh:
                det.activate(self.frame_id)
                activated.append(det)

        # ── Remove long-lost tracks ──────────────────────────────────────────
        for t in self.lost_stracks:
            if self.frame_id - t.frame_id > self.track_buffer:
                t.mark_removed()
                removed.append(t)

        # ── Update state lists ───────────────────────────────────────────────
        self.tracked_stracks = [t for t in self.tracked_stracks
                                if t.state == TrackState.Tracked]
        self.tracked_stracks = _merge_lists(self.tracked_stracks, activated)
        self.tracked_stracks = _merge_lists(self.tracked_stracks, refound)
        self.lost_stracks = _sub_lists(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = _sub_lists(self.lost_stracks, removed)
        self.removed_stracks.extend(removed)
        # keep removed list bounded
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-500:]

        # ── Deflate bboxes back to original size for output ────────────────
        result = [t for t in self.tracked_stracks if t.is_activated]
        if inflate:
            for t in result:
                t.tlbr = self._deflate_bbox(t.tlbr, self.bbox_inflate)
        return result

    def reset(self):
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        STrack.reset_id()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _merge_lists(a: List[STrack], b: List[STrack]) -> List[STrack]:
    ids = set(t.track_id for t in a)
    merged = list(a)
    for t in b:
        if t.track_id not in ids:
            merged.append(t)
            ids.add(t.track_id)
    return merged


def _sub_lists(a: List[STrack], b: List[STrack]) -> List[STrack]:
    ids = set(t.track_id for t in b)
    return [t for t in a if t.track_id not in ids]
