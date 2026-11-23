"""Object detection with a pre-trained YOLOv4-tiny model through OpenCV's DNN module."""
import os

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(BASE_DIR, "models", "yolov4-tiny.cfg")
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "yolov4-tiny.weights")
NAMES_PATH = os.path.join(BASE_DIR, "models", "coco.names")


class YOLODetector:
    def __init__(self, conf_threshold=0.5, nms_threshold=0.4, input_size=416, allowed_classes=None):
        for path in (CFG_PATH, WEIGHTS_PATH, NAMES_PATH):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model file missing: {path}\nRun  python download_model.py  first.")

        with open(NAMES_PATH, "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(input_size, input_size), scale=1 / 255.0, swapRB=True)

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.allowed_ids = None
        if allowed_classes:
            wanted = {c.lower() for c in allowed_classes}
            self.allowed_ids = {i for i, n in enumerate(self.class_names) if n.lower() in wanted}
            unknown = wanted - {self.class_names[i].lower() for i in self.allowed_ids}
            if unknown:
                raise ValueError(f"Unknown class name(s): {sorted(unknown)}. "
                                 f"Valid names are listed in models/coco.names")

    def detect(self, frame):
        """Return an array of detections: [x1, y1, x2, y2, confidence, class_id]."""
        class_ids, scores, boxes = self.model.detect(
            frame, confThreshold=self.conf_threshold, nmsThreshold=self.nms_threshold)
        class_ids = np.asarray(class_ids).reshape(-1)
        scores = np.asarray(scores).reshape(-1)
        boxes = np.asarray(boxes).reshape(-1, 4)

        h, w = frame.shape[:2]
        dets = []
        for cid, score, (x, y, bw, bh) in zip(class_ids, scores, boxes):
            if self.allowed_ids is not None and int(cid) not in self.allowed_ids:
                continue
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w - 1, x + bw), min(h - 1, y + bh)
            if x2 > x1 and y2 > y1:
                dets.append([x1, y1, x2, y2, float(score), int(cid)])
        return np.array(dets, dtype=np.float64).reshape(-1, 6)
