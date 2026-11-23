"""SORT - Simple Online and Realtime Tracking (Bewley et al., 2016).

Each tracked object has a Kalman filter that predicts its box in the next frame.
New detections are matched to the predicted boxes with the Hungarian algorithm
using IoU (overlap) as the cost.  Every object keeps the same ID while it is tracked.
"""
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # scipy missing -> fall back to a simple greedy matcher
    linear_sum_assignment = None


def iou_batch(a, b):
    """IoU between every box in a (N,4) and every box in b (M,4). Boxes are x1,y1,x2,y2."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _box_to_z(box):
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2.0, box[1] + h / 2.0, w * h, w / max(h, 1e-6)]).reshape(4, 1)


def _x_to_box(x):
    s, r = max(float(x[2, 0]), 1e-6), max(float(x[3, 0]), 1e-6)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return np.array([x[0, 0] - w / 2.0, x[1, 0] - h / 2.0, x[0, 0] + w / 2.0, x[1, 0] + h / 2.0])


class KalmanBoxTracker:
    """Tracks one object. State = [cx, cy, area, aspect, vx, vy, v_area]."""
    count = 0

    def __init__(self, det):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4])
        self.x = np.zeros((7, 1))
        self.x[:4] = _box_to_z(det[:4])

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.score = float(det[4])
        self.class_id = int(det[5])
        self.class_votes = {self.class_id: float(det[4])}

    def predict(self):
        if self.x[2, 0] + self.x[6, 0] <= 0:      # keep area positive
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _x_to_box(self.x)

    def update(self, det):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = float(det[4])
        # class = the label with the highest total confidence over time (stable labels)
        cid = int(det[5])
        self.class_votes[cid] = self.class_votes.get(cid, 0.0) + float(det[4])
        self.class_id = max(self.class_votes, key=self.class_votes.get)

        z = _box_to_z(det[:4])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def box(self):
        return _x_to_box(self.x)


def _match(iou_matrix, threshold):
    """Hungarian matching. Returns matches (list of (det_idx, trk_idx))."""
    if iou_matrix.size == 0:
        return []
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(-iou_matrix)
        pairs = list(zip(rows, cols))
    else:  # greedy fallback
        pairs, used_r, used_c = [], set(), set()
        order = np.dstack(np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape))[0]
        for r, c in order:
            if r not in used_r and c not in used_c:
                pairs.append((r, c))
                used_r.add(r)
                used_c.add(c)
    return [(int(r), int(c)) for r, c in pairs if iou_matrix[r, c] >= threshold]


class Sort:
    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, detections):
        """detections: (N,6) [x1,y1,x2,y2,score,class_id].
        Returns (K,7) array: [x1,y1,x2,y2,track_id,class_id,score]."""
        self.frame_count += 1
        detections = np.asarray(detections, dtype=np.float64).reshape(-1, 6)

        # 1) predict new positions of existing tracks
        predicted = []
        for t in self.trackers:
            box = t.predict()
            predicted.append(box)
        valid = [i for i, b in enumerate(predicted) if np.all(np.isfinite(b))]
        self.trackers = [self.trackers[i] for i in valid]
        predicted = [predicted[i] for i in valid]

        # 2) match detections <-> tracks
        iou = iou_batch(detections[:, :4], np.array(predicted).reshape(-1, 4))
        matches = _match(iou, self.iou_threshold)
        matched_d = {d for d, _ in matches}

        for d, t in matches:
            self.trackers[t].update(detections[d])

        # 3) new tracks for unmatched detections
        for d in range(len(detections)):
            if d not in matched_d:
                self.trackers.append(KalmanBoxTracker(detections[d]))

        # 4) collect results and drop dead tracks
        results = []
        for t in self.trackers:
            if t.time_since_update < 1 and (t.hit_streak >= self.min_hits
                                            or self.frame_count <= self.min_hits):
                box = t.box()
                results.append([*box, t.id, t.class_id, t.score])
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return np.array(results, dtype=np.float64).reshape(-1, 7)
