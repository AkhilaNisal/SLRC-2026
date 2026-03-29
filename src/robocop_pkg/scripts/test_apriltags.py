#!/usr/bin/env python3
"""
Standalone AprilTag test — no ROS, no robot movement.

Opens the camera directly, runs the pupil_apriltags detector, and shows a
live debug window with detected tag IDs drawn on the frame.

Usage:
    python3 scripts/test_apriltags.py
    python3 scripts/test_apriltags.py --camera 1
    python3 scripts/test_apriltags.py --family tag36h11
    python3 scripts/test_apriltags.py --camera 0 --width 1280 --height 720

Press Q or Esc to quit.
"""

import argparse
import sys

import cv2
from pupil_apriltags import Detector


def main():
    parser = argparse.ArgumentParser(description='AprilTag live detection test')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=640, help='Capture width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Capture height (default: 480)')
    parser.add_argument('--fps', type=int, default=30, help='Capture FPS (default: 30)')
    parser.add_argument('--family', type=str, default='tagStandard52h13',
                        help='AprilTag family (default: tagStandard52h13)')
    parser.add_argument('--decimate', type=float, default=1.0,
                        help='Quad decimate factor — higher = faster but less sensitive (default: 1.0)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open camera index {args.camera}. Try --camera 1', file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    detector = Detector(
        families=args.family,
        nthreads=2,
        quad_decimate=args.decimate,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    print(f'[INFO] Camera {args.camera} opened — {args.width}x{args.height} @ {args.fps}fps')
    print(f'[INFO] Detecting family: {args.family}')
    print('[INFO] Press Q or Esc to quit.')

    seen_ids: set = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print('[WARN] Failed to grab frame', file=sys.stderr)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray)

        for tag in detections:
            tag_id = int(tag.tag_id)
            corners = tag.corners.reshape((-1, 1, 2)).astype(int)
            center = (int(tag.center[0]), int(tag.center[1]))

            cv2.polylines(frame, [corners], True, (0, 0, 255), 3)
            cv2.circle(frame, center, 6, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f'ID:{tag_id}',
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            if tag_id not in seen_ids:
                seen_ids.add(tag_id)
                print(f'[NEW TAG] ID={tag_id}  total unique={len(seen_ids)}  all={sorted(seen_ids)}')

        # HUD
        cv2.putText(
            frame,
            f'Detected: {len(detections)}  Unique: {len(seen_ids)}',
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )

        cv2.imshow('AprilTag Test', frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q or Esc
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f'[INFO] Done. Unique tags seen: {sorted(seen_ids)}')


if __name__ == '__main__':
    main()
