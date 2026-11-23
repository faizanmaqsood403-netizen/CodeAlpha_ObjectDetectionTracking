"""Object Detection and Tracking - YOLO (detection) + SORT (tracking) with OpenCV.

Examples
    python main.py                                   # webcam
    python main.py --source data/sample_video.avi    # video file
    python main.py --source video.mp4 --save         # also save the result video
    python main.py --classes person car              # only detect/track these classes

Keys in the window:  Q or ESC = quit   P = pause/resume   S = save screenshot
"""
import argparse
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime

import cv2
import numpy as np

from detector import YOLODetector
from sort import Sort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
WINDOW = "Object Detection and Tracking (Q = quit, P = pause, S = screenshot)"


def parse_args():
    p = argparse.ArgumentParser(description="Real-time object detection and tracking (YOLO + SORT)")
    p.add_argument("--source", default="0",
                   help="0 = default webcam (1, 2 ... other cameras) or path to a video file")
    p.add_argument("--conf", type=float, default=0.5, help="detection confidence threshold")
    p.add_argument("--classes", nargs="*", default=None,
                   help="only keep these classes, e.g. --classes person car")
    p.add_argument("--save", action="store_true", help="save the output video to output/")
    p.add_argument("--no-display", action="store_true", help="do not open a window")
    p.add_argument("--width", type=int, default=960,
                   help="resize frames wider than this (smaller = faster)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    p.add_argument("--no-trails", action="store_true", help="do not draw movement trails")
    return p.parse_args()


def open_source(source: str):
    """Open a webcam index or a video file."""
    if source.isdigit():
        index = int(source)
        cap = None
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)   # most reliable on Windows
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam {index}. Close other apps that use the camera, "
                f"try --source 1, or use a video file:  python main.py --source data/sample_video.avi")
        return cap, True
    if not os.path.exists(source):
        raise FileNotFoundError(f"Video file not found: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not read video file: {source}")
    return cap, False


def color_for_id(track_id: int):
    """A bright, stable colour for every tracking ID (BGR)."""
    rng = np.random.RandomState(int(track_id) * 7919 % (2 ** 31))
    c = rng.randint(60, 256, 3)
    return int(c[0]), int(c[1]), int(c[2])


def draw_label(img, text, x, y, color):
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x = max(0, min(x, img.shape[1] - tw - 8))          # keep the label inside the image
    y_top = max(y - th - base - 4, 0)
    cv2.rectangle(img, (x, y_top), (x + tw + 6, y_top + th + base + 4), color, -1)
    cv2.putText(img, text, (x + 3, y_top + th + 1), font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def draw_hud(img, fps, n_objects, class_counts):
    lines = [f"FPS: {fps:4.1f}", f"Tracked objects: {n_objects}"]
    lines += [f"{name}: {cnt}" for name, cnt in sorted(class_counts.items())][:6]
    h = 12 + 22 * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (230, 8 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (16, 30 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if i < 2 else (255, 255, 255), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        detector = YOLODetector(conf_threshold=args.conf, allowed_classes=args.classes)
        cap, is_webcam = open_source(args.source)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        sys.exit(f"\nERROR: {exc}")

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    trails = defaultdict(lambda: deque(maxlen=30))
    seen_ids = defaultdict(set)                      # class name -> set of track ids

    show = not args.no_display
    writer = None
    save_path = None
    frame_idx, fps_smooth, paused = 0, 0.0, False
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if src_fps and 1 < src_fps < 120 else 25.0
    started = time.time()
    print("[main] Running ... press Q (or ESC) in the video window to stop.")

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_idx += 1
                t0 = time.time()

                h, w = frame.shape[:2]
                if w > args.width:
                    frame = cv2.resize(frame, (args.width, int(h * args.width / w)))

                # 1) detect   2) track   3) draw
                detections = detector.detect(frame)
                tracks = tracker.update(detections)

                class_counts = defaultdict(int)
                for x1, y1, x2, y2, tid, cid, score in tracks:
                    tid, cid = int(tid), int(cid)
                    name = detector.class_names[cid] if cid < len(detector.class_names) else str(cid)
                    color = color_for_id(tid)
                    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                    cv2.rectangle(frame, p1, p2, color, 2)
                    draw_label(frame, f"ID {tid} {name} {score:.2f}", p1[0], p1[1], color)
                    class_counts[name] += 1
                    seen_ids[name].add(tid)
                    if not args.no_trails:
                        trails[tid].append((int((x1 + x2) / 2), int(y2)))
                        pts = list(trails[tid])
                        for a, b in zip(pts[:-1], pts[1:]):
                            cv2.line(frame, a, b, color, 2)

                dt = max(time.time() - t0, 1e-6)
                fps_smooth = (1.0 / dt) if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * (1.0 / dt)
                draw_hud(frame, fps_smooth, len(tracks), class_counts)

                if args.save:
                    if writer is None:
                        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = os.path.join(OUTPUT_DIR, f"tracked_{stamp}.mp4")
                        fh, fw = frame.shape[:2]
                        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 src_fps, (fw, fh))
                        if not writer.isOpened():
                            print("[main] WARNING: could not create the output video; continuing without saving.")
                            writer, args.save = None, False
                    if writer is not None:
                        writer.write(frame)

                if args.max_frames and frame_idx >= args.max_frames:
                    break
                if frame_idx % 50 == 0:
                    print(f"[main] frame {frame_idx} | {fps_smooth:4.1f} FPS | "
                          f"{len(tracks)} objects tracked")

            if show:
                try:
                    cv2.imshow(WINDOW, frame)
                    key = cv2.waitKey(1 if not paused else 50) & 0xFF
                except cv2.error:
                    print("[main] This OpenCV build has no window support -> running without display.\n"
                          "       Fix: pip uninstall opencv-python-headless -y && pip install opencv-python")
                    show = False
                    continue
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("p"), ord("P")):
                    paused = not paused
                if key in (ord("s"), ord("S")):
                    shot = os.path.join(OUTPUT_DIR, f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.png")
                    cv2.imwrite(shot, frame)
                    print(f"[main] Screenshot saved: {shot}")
                if frame_idx > 5:                           # window closed with the X button
                    try:
                        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except cv2.error:
                        pass
    except KeyboardInterrupt:
        print("\n[main] Stopped by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    elapsed = max(time.time() - started, 1e-6)
    print("\n================ SUMMARY ================")
    print(f"Frames processed : {frame_idx}")
    print(f"Average speed    : {frame_idx / elapsed:.1f} FPS")
    total = sum(len(v) for v in seen_ids.values())
    print(f"Unique objects   : {total}")
    for name, ids in sorted(seen_ids.items(), key=lambda kv: -len(kv[1])):
        print(f"   {name:<15}: {len(ids)}")
    if save_path:
        print(f"Saved video      : {save_path}")
    print("=========================================")


if __name__ == "__main__":
    main()
