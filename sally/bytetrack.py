"""
ByteTrack — multi-object tracker with bounding box doubling.

Inflate bboxes before tracking (better IoU for small overhead targets),
deflate back to real size after. Set bbox_inflate=1.0 to disable.
"""

import numpy as np
from typing import List, Tuple, Optional


# ── Kalman Filter ────────────────────────────────────────────────────────────
class _KalmanFilter:
    _W_POS = 1.0 / 20
    _W_VEL = 1.0 / 160

    def __init__(self):
        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.H = np.eye(4, 8, dtype=np.float64)

    def initiate(self, m):
        mean = np.zeros(8, dtype=np.float64)
        mean[:4] = m
        s = [2*self._W_POS*m[3], 2*self._W_POS*m[3], 1e-2, 2*self._W_POS*m[3],
             10*self._W_VEL*m[3], 10*self._W_VEL*m[3], 1e-5, 10*self._W_VEL*m[3]]
        return mean, np.diag(np.square(s))

    def predict(self, mean, cov):
        s = [self._W_POS*mean[3], self._W_POS*mean[3], 1e-2, self._W_POS*mean[3],
             self._W_VEL*mean[3], self._W_VEL*mean[3], 1e-5, self._W_VEL*mean[3]]
        Q = np.diag(np.square(s))
        return self.F @ mean, self.F @ cov @ self.F.T + Q

    def update(self, mean, cov, measurement):
        s = [self._W_POS*mean[3], self._W_POS*mean[3], 1e-2, self._W_POS*mean[3]]
        R = np.diag(np.square(s))
        S = self.H @ cov @ self.H.T + R
        K = cov @ self.H.T @ np.linalg.inv(S)
        mean = mean + K @ (measurement - self.H @ mean)
        cov = (np.eye(8) - K @ self.H) @ cov
        return mean, cov


_kf = _KalmanFilter()


# ── Track ────────────────────────────────────────────────────────────────────
class STrack:
    _next_id = 1
    TRACKED, LOST, REMOVED = 1, 2, 3

    def __init__(self, tlbr, score):
        self.track_id = 0
        self.tlbr = np.asarray(tlbr, dtype=np.float32).copy()
        self.score = float(score)
        self.state = 0
        self.is_activated = False
        self.frame_id = 0
        self._mean = self._cov = None

    @staticmethod
    def _to_xyah(tlbr):
        w, h = tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]
        return np.array([tlbr[0] + w/2, tlbr[1] + h/2, w/(h+1e-6), h], dtype=np.float64)

    @staticmethod
    def _to_tlbr(xyah):
        cx, cy, a, h = xyah
        w = a * h
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dtype=np.float32)

    def activate(self, fid):
        self.track_id = STrack._next_id; STrack._next_id += 1
        self._mean, self._cov = _kf.initiate(self._to_xyah(self.tlbr))
        self.state = self.TRACKED; self.is_activated = True; self.frame_id = fid

    def predict(self):
        if self._mean is not None:
            self._mean, self._cov = _kf.predict(self._mean, self._cov)
            self.tlbr = self._to_tlbr(self._mean[:4])

    def update(self, det, fid):
        m = self._to_xyah(det.tlbr)
        self._mean, self._cov = _kf.update(self._mean, self._cov, m)
        self.tlbr = self._to_tlbr(self._mean[:4])
        self.score = det.score
        self.state = self.TRACKED; self.is_activated = True; self.frame_id = fid

    def reactivate(self, det, fid):
        self.update(det, fid)

    @staticmethod
    def reset_id():
        STrack._next_id = 1


# ── IoU ──────────────────────────────────────────────────────────────────────
def _iou_batch(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(br - tl, 0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1])
    area_b = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


def _match(cost, thresh):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = cost.shape
    ma, mb, matches = set(), set(), []
    for idx in np.argsort(cost.ravel()):
        r, c = divmod(int(idx), cols)
        if cost[r, c] > thresh:
            break
        if r not in ma and c not in mb:
            matches.append((r, c)); ma.add(r); mb.add(c)
            if len(ma) == rows or len(mb) == cols:
                break
    return matches, [i for i in range(rows) if i not in ma], [j for j in range(cols) if j not in mb]


# ── ByteTracker ──────────────────────────────────────────────────────────────
class ByteTracker:
    def __init__(self, track_thresh=0.45, match_thresh=0.8,
                 track_buffer=30, low_thresh=0.1, bbox_inflate=1.0):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.low_thresh = low_thresh
        self.bbox_inflate = bbox_inflate
        self.tracked: List[STrack] = []
        self.lost: List[STrack] = []
        self.frame_id = 0

    @staticmethod
    def _inflate(tlbr, f):
        cx, cy = (tlbr[0]+tlbr[2])/2, (tlbr[1]+tlbr[3])/2
        hw, hh = (tlbr[2]-tlbr[0])/2*f, (tlbr[3]-tlbr[1])/2*f
        return np.array([cx-hw, cy-hh, cx+hw, cy+hh], dtype=np.float32)

    def update(self, dets: List[dict]) -> List[STrack]:
        self.frame_id += 1
        inflate = self.bbox_inflate > 1.0

        if dets:
            boxes_orig = np.array([d["bbox"] for d in dets], dtype=np.float32)
            scores = np.array([d["conf"] for d in dets], dtype=np.float32)
            boxes = (np.array([self._inflate(b, self.bbox_inflate) for b in boxes_orig])
                     if inflate else boxes_orig)
        else:
            boxes = np.empty((0, 4), np.float32)
            boxes_orig = boxes
            scores = np.empty(0, np.float32)

        hi = scores >= self.track_thresh
        lo = (~hi) & (scores >= self.low_thresh)
        hi_dets = [STrack(boxes[i], scores[i]) for i in np.where(hi)[0]]
        lo_dets = [STrack(boxes[i], scores[i]) for i in np.where(lo)[0]]

        pool = self.tracked + self.lost
        for t in pool:
            t.predict()

        active = [t for t in self.tracked if t.state == STrack.TRACKED]
        activated, refound, new_lost = [], [], []

        # Stage 1: high-conf vs active
        if active and hi_dets:
            cost = 1 - _iou_batch([t.tlbr for t in active], [d.tlbr for d in hi_dets])
            m1, ut1, ud1 = _match(cost, self.match_thresh)
        else:
            m1, ut1, ud1 = [], list(range(len(active))), list(range(len(hi_dets)))
        for ti, di in m1:
            active[ti].update(hi_dets[di], self.frame_id); activated.append(active[ti])

        # Stage 2: low-conf vs remaining active
        rem = [active[i] for i in ut1]
        if rem and lo_dets:
            cost = 1 - _iou_batch([t.tlbr for t in rem], [d.tlbr for d in lo_dets])
            m2, ut2, _ = _match(cost, self.match_thresh)
        else:
            m2, ut2 = [], list(range(len(rem)))
        for ti, di in m2:
            rem[ti].update(lo_dets[di], self.frame_id); activated.append(rem[ti])
        for i in ut2:
            rem[i].state = STrack.LOST; new_lost.append(rem[i])

        # Stage 3: unmatched high-conf vs lost
        umhi = [hi_dets[i] for i in ud1]
        if self.lost and umhi:
            cost = 1 - _iou_batch([t.tlbr for t in self.lost], [d.tlbr for d in umhi])
            m3, _, ud3 = _match(cost, self.match_thresh)
        else:
            m3, ud3 = [], list(range(len(umhi)))
        for ti, di in m3:
            self.lost[ti].reactivate(umhi[di], self.frame_id); refound.append(self.lost[ti])

        # New tracks
        for i in ud3:
            if umhi[i].score >= self.track_thresh:
                umhi[i].activate(self.frame_id); activated.append(umhi[i])

        # Cleanup
        removed_ids = set()
        for t in self.lost:
            if self.frame_id - t.frame_id > self.track_buffer:
                t.state = STrack.REMOVED; removed_ids.add(t.track_id)

        ids = set()
        merged = []
        for t in self.tracked:
            if t.state == STrack.TRACKED and t.track_id not in ids:
                merged.append(t); ids.add(t.track_id)
        for t in activated + refound:
            if t.track_id not in ids:
                merged.append(t); ids.add(t.track_id)
        self.tracked = merged

        lost_ids = set(t.track_id for t in self.tracked)
        self.lost = [t for t in self.lost if t.track_id not in lost_ids and t.track_id not in removed_ids]
        self.lost.extend(new_lost)

        result = [t for t in self.tracked if t.is_activated]

        # Deflate back to original size
        if inflate:
            for t in result:
                t.tlbr = self._inflate(t.tlbr, 1.0 / self.bbox_inflate)

        return result

    def reset(self):
        self.tracked.clear(); self.lost.clear(); self.frame_id = 0; STrack.reset_id()
