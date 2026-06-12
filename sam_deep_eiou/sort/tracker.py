"""SORT tracker ported from https://github.com/abewley/sort

Modifications:
- Replaced filterpy dependency with inline Kalman filter
- Added assignment margin computation for SAM-Deep-EIoU integration
- Added edge-kill (tracks at frame border are removed instead of lost)
- update() returns tracker objects with .track_id, .last_tlbr, .margin
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman_filter import KalmanFilter
from ..utils.shared_constants import BORDER_MARGIN


def _linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """Computes IoU between two sets of bboxes [x1,y1,x2,y2]."""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


def convert_bbox_to_z(bbox):
    """[x1,y1,x2,y2] -> [x,y,s,r] (center, area, aspect ratio)."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """[x,y,s,r] -> [x1,y1,x2,y2]."""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2., score]).reshape((1, 5))


class KalmanBoxTracker:
    """Tracked object state using Kalman filter."""
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]], dtype=float)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]], dtype=float)
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.margin = float('inf')

    @property
    def track_id(self):
        return self.id + 1

    @property
    def last_tlbr(self):
        return self.get_state()[0]

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """Assigns detections to tracked objects.

    Returns: matches, unmatched_detections, unmatched_trackers, iou_matrix
    """
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int), np.arange(len(detections)),
                np.empty((0, 5), dtype=int), np.empty((0, 0)))

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = _linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers), iou_matrix


class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3,
                 frame_width=None, frame_height=None):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.frame_width = frame_width
        self.frame_height = frame_height

    def _bbox_at_border(self, tlbr):
        if self.frame_width is None or self.frame_height is None:
            return False
        x1, y1, x2, y2 = tlbr
        m = BORDER_MARGIN
        return x1 <= m or y1 <= m or x2 >= self.frame_width - 1 - m or y2 >= self.frame_height - 1 - m

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
            dets: numpy array of detections [[x1,y1,x2,y2,score],...]
        Returns:
            list of active KalmanBoxTracker objects with .track_id, .last_tlbr, .margin
        """
        self.frame_count += 1

        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks, iou_matrix = \
            associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Update matched trackers with assigned detections + compute margin
        for m in matched:
            d_idx, t_idx = m[0], m[1]
            self.trackers[t_idx].update(dets[d_idx, :])

            # Margin: gap between best and second-best IoU for this detection
            if iou_matrix.size > 0 and iou_matrix.shape[1] > 1:
                cost_row = -iou_matrix[d_idx, :]
                sorted_row = np.sort(cost_row)
                self.trackers[t_idx].margin = sorted_row[1] - sorted_row[0]
            else:
                self.trackers[t_idx].margin = float('inf')

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        # Collect results and remove dead/border trackers
        i = len(self.trackers)
        ret = []
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                continue
            # Edge-kill: unmatched track at frame border
            if trk.time_since_update > 0 and self._bbox_at_border(trk.last_tlbr):
                self.trackers.pop(i)
                continue
            if (trk.time_since_update < 1) and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(trk)

        return ret
