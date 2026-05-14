# SERS

Smart Emergency Response System accident detection using YOLO on video frames.

This project is set up to train and run a YOLO model that detects road accidents
from CCTV, dashcam, or uploaded video. The pipeline is:

1. Extract frames from video.
2. Annotate accident regions in YOLO format.
3. Train a custom YOLO model.
4. Run the trained model on video and trigger alerts only after repeated
   accident detections across consecutive frames.

## Project Layout

```text
SERS/
  data/
    sers_accident.yaml        # YOLO dataset config
  src/sers_yolo/
    extract_frames.py         # Create training images from videos
    train.py                  # Train YOLO accident detector
    detect_video.py           # Run detection on videos
  requirements.txt
```

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Prepare Dataset

Extract frames from your accident and normal traffic videos:

```powershell
python -m src.sers_yolo.extract_frames --video path\to\traffic.mp4 --out datasets\raw_frames --every 10
```

Annotate the extracted frames with a tool such as CVAT, Label Studio, or
Roboflow, then export in YOLO format. The expected dataset structure is:

```text
datasets/accident_yolo/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Use one class named `accident`.

## Train

```powershell
python -m src.sers_yolo.train --data data\sers_accident.yaml --model yolov8n.pt --epochs 50 --imgsz 640
```

The trained weights will be saved under:

```text
runs/detect/sers_accident/weights/best.pt
```

## Detect Accidents In Video

```powershell
python -m src.sers_yolo.detect_video --weights runs\detect\sers_accident\weights\best.pt --source path\to\video.mp4 --output outputs\detected.mp4 --alert-log outputs\alerts.csv --alert-jsonl outputs\alerts.jsonl
```

Useful options:

```powershell
--conf 0.35              # confidence threshold
--window 12              # number of recent frames to inspect
--trigger-frames 4       # accident frames needed inside the window
--accident-class accident
--camera-id CAM-BLR-042  # included in SERS alert payloads
```

For real-time camera input:

```powershell
python -m src.sers_yolo.detect_video --weights runs\detect\sers_accident\weights\best.pt --source 0
```

## Notes

- A general YOLO model cannot reliably detect accidents without custom training
  data. You need labeled accident frames for this project.
- Accident detection works best when you include both positive accident frames
  and negative normal-traffic frames from similar camera angles.
- For SERS integration, use `outputs/alerts.csv` or the console `ALERT` events
  to trigger emergency notification, GPS/location lookup, and response routing.
- Use `outputs/alerts.jsonl` when you need proposal-style structured payloads
  with event type, severity, confidence, timestamp, camera ID, and dispatch flag.
