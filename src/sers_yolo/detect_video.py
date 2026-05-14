from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def accident_class_ids(names: dict[int, str], accident_class: str | None) -> set[int]:
    if accident_class:
        wanted = {name.strip().lower() for name in accident_class.split(",")}
        return {idx for idx, name in names.items() if name.lower() in wanted}

    keywords = ("accident", "crash", "collision", "wreck")
    return {idx for idx, name in names.items() if any(key in name.lower() for key in keywords)}


def accident_confidence(result, class_ids: set[int]) -> float:
    if result.boxes is None or len(result.boxes) == 0:
        return 0.0

    best = 0.0
    for cls, conf in zip(result.boxes.cls, result.boxes.conf):
        if int(cls.item()) in class_ids:
            best = max(best, float(conf.item()))

    return best


def draw_alert_banner(frame, text: str) -> None:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 48), (0, 0, 180), thickness=-1)
    cv2.putText(
        frame,
        text,
        (14, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def severity_from_confidence(confidence: float) -> str:
    if confidence >= 0.85:
        return "critical"
    if confidence >= 0.65:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


def write_alert(
    csv_writer: csv.writer | None,
    jsonl_file,
    frame_index: int,
    timestamp_s: float,
    score: int,
    confidence: float,
    camera_id: str,
) -> None:
    severity = severity_from_confidence(confidence)
    message = (
        f"ALERT frame={frame_index} time={timestamp_s:.2f}s "
        f"votes={score} confidence={confidence:.2f} severity={severity}"
    )
    print(message)
    if csv_writer:
        csv_writer.writerow([frame_index, f"{timestamp_s:.2f}", score, f"{confidence:.4f}", severity])
    if jsonl_file:
        payload = {
            "event_type": "collision",
            "severity": severity,
            "confidence": round(confidence, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "camera_id": camera_id,
            "frame": frame_index,
            "video_timestamp_seconds": round(timestamp_s, 2),
            "dispatch_triggered": True,
        }
        jsonl_file.write(json.dumps(payload) + "\n")
        jsonl_file.flush()


def run_detection(
    weights: str,
    source: str,
    output: str | None,
    alert_log: str | None,
    alert_jsonl: str | None,
    camera_id: str,
    conf: float,
    imgsz: int,
    window: int,
    trigger_frames: int,
    accident_class: str | None,
    show: bool,
) -> None:
    model = YOLO(weights)
    class_ids = accident_class_ids(model.names, accident_class)
    if not class_ids:
        raise ValueError(
            "No accident class found. Use --accident-class with the class name in your model, "
            "for example --accident-class accident."
        )

    capture = cv2.VideoCapture(parse_source(source))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video source: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = None
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    csv_file = None
    csv_writer = None
    if alert_log:
        Path(alert_log).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(alert_log, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "timestamp_seconds", "accident_votes_in_window", "confidence", "severity"])

    jsonl_file = None
    if alert_jsonl:
        Path(alert_jsonl).parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = open(alert_jsonl, "w", encoding="utf-8")

    votes: deque[bool] = deque(maxlen=window)
    last_alert_at = 0.0
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
            confidence = accident_confidence(result, class_ids)
            votes.append(confidence >= conf)

            rendered = result.plot()
            vote_count = sum(votes)
            alert_active = len(votes) == window and vote_count >= trigger_frames

            if alert_active:
                draw_alert_banner(rendered, f"ACCIDENT DETECTED ({vote_count}/{window})")
                now = time.monotonic()
                if now - last_alert_at > 5:
                    write_alert(
                        csv_writer,
                        jsonl_file,
                        frame_index,
                        frame_index / fps,
                        vote_count,
                        confidence,
                        camera_id,
                    )
                    last_alert_at = now

            if video_writer:
                video_writer.write(rendered)

            if show:
                cv2.imshow("SERS Accident Detection", rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        capture.release()
        if video_writer:
            video_writer.release()
        if csv_file:
            csv_file.close()
        if jsonl_file:
            jsonl_file.close()
        if show:
            cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO accident detection on a video stream.")
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights, e.g. best.pt.")
    parser.add_argument("--source", required=True, help="Video path, RTSP URL, or camera index like 0.")
    parser.add_argument("--output", default=None, help="Optional annotated output video path.")
    parser.add_argument("--alert-log", default=None, help="Optional CSV path for accident alert events.")
    parser.add_argument("--alert-jsonl", default=None, help="Optional SERS JSONL alert payload path.")
    parser.add_argument("--camera-id", default="CAM-SERS-001", help="Camera ID to include in alerts.")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--window", type=int, default=12, help="Recent frame window for alert smoothing.")
    parser.add_argument("--trigger-frames", type=int, default=4, help="Accident votes needed inside window.")
    parser.add_argument("--accident-class", default=None, help="Comma-separated accident class names.")
    parser.add_argument("--show", action="store_true", help="Display live preview. Press q to exit.")
    return parser


def validate_window(window: int, trigger_frames: int) -> None:
    if window < 1:
        raise ValueError("--window must be 1 or greater")
    if trigger_frames < 1 or trigger_frames > window:
        raise ValueError("--trigger-frames must be between 1 and --window")


def main() -> None:
    args = build_parser().parse_args()
    validate_window(args.window, args.trigger_frames)
    run_detection(
        weights=args.weights,
        source=args.source,
        output=args.output,
        alert_log=args.alert_log,
        alert_jsonl=args.alert_jsonl,
        camera_id=args.camera_id,
        conf=args.conf,
        imgsz=args.imgsz,
        window=args.window,
        trigger_frames=args.trigger_frames,
        accident_class=args.accident_class,
        show=args.show,
    )


if __name__ == "__main__":
    main()
