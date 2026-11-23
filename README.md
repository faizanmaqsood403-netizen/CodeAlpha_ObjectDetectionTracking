# CodeAlpha_Object_Detection_Tracking

AI internship — **Task 4: Object Detection and Tracking** (CodeAlpha)

Real-time object detection with a pre-trained **YOLO** model and multi-object tracking with **SORT**,
using **OpenCV**. Every object gets a bounding box, a class label and a unique tracking ID, plus a movement trail.

## How it works (maps to the task requirements)
| Task requirement | Where |
|---|---|
| Real-time video input (webcam or video file) with OpenCV | `main.py` (`--source 0` = webcam, or a video path) |
| Pre-trained model like YOLO | `detector.py` - YOLOv4-tiny (COCO, 80 classes) loaded with OpenCV DNN |
| Process each frame, detect objects, draw bounding boxes | `main.py` + `detector.py` |
| Object tracking with SORT / Deep SORT | `sort.py` - SORT (Kalman filter + Hungarian matching on IoU) |
| Display output with labels and tracking IDs in real time | `main.py` shows `ID <n> <class> <confidence>`, trails, FPS and counts |

## Project structure
```
CodeAlpha_Object_Detection_Tracking/
├── main.py              # run this: video input -> detect -> track -> display/save
├── detector.py          # YOLOv4-tiny detector (OpenCV DNN)
├── sort.py              # SORT tracker (Kalman filter + Hungarian algorithm)
├── download_model.py    # re-downloads model + sample video if they are missing (resumable)
├── requirements.txt
├── models/              # yolov4-tiny.cfg / .weights / coco.names
├── data/sample_video.avi
├── output/              # your saved videos and screenshots
└── sample_output/       # example result (video + screenshot)
```

## Setup
Python 3.9 - 3.12.
```bash
pip install -r requirements.txt
```
(The model files and a sample video are already included. If they are ever missing, run `python download_model.py`.)

## Run
```bash
python main.py                                    # webcam
python main.py --source data/sample_video.avi     # sample video
python main.py --source my_video.mp4 --save       # your video, save result to output/
python main.py --classes person car               # only these classes
python main.py --width 640                        # smaller frames = faster
python main.py --conf 0.4                         # lower = more detections
```
Keys in the window: **Q / ESC** quit, **P** pause, **S** save a screenshot to `output/`.
When it finishes, a summary of unique tracked objects per class is printed.

## Notes
* Uses OpenCV 4.x (`opencv-python>=4.8,<5`) because OpenCV 5 removed the Darknet (YOLO) loader.
* Runs on CPU, no GPU needed. YOLOv4-tiny is small and fast; larger models are more accurate but slower.
* An object keeps its ID while visible; if it leaves the frame for a long time it may get a new ID (normal for SORT).

## Tech
Python, OpenCV (DNN), NumPy, SciPy, YOLOv4-tiny, SORT
