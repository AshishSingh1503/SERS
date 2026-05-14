from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frames(video: str, out: str, every: int, prefix: str | None) -> int:
    video_path = Path(video)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    stem = prefix or video_path.stem
    frame_index = 0
    saved = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % every == 0:
            target = out_dir / f"{stem}_{frame_index:06d}.jpg"
            cv2.imwrite(str(target), frame)
            saved += 1

        frame_index += 1

    capture.release()
    return saved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract video frames for YOLO annotation.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--out", required=True, help="Output folder for extracted JPG frames.")
    parser.add_argument("--every", type=int, default=10, help="Save one frame every N frames.")
    parser.add_argument("--prefix", default=None, help="Optional filename prefix.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.every < 1:
        raise ValueError("--every must be 1 or greater")

    saved = extract_frames(args.video, args.out, args.every, args.prefix)
    print(f"Saved {saved} frames to {args.out}")


if __name__ == "__main__":
    main()
