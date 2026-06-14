#!/usr/bin/env python3
"""
Standalone ROI + white-line detection tester.
No ROS2 needed. Uses same logic as line_detection_utils.py.

Controls:
  Trackbars - tune HSV thresholds and ROI start live
  q          - quit
  s          - save current frame + mask to /tmp/
"""
import cv2
import numpy as np
import sys
import time

CAMERA_INDEX = 0
WIN_CAM = 'Camera (ROI overlay)'
WIN_MASK = 'White Mask'
WIN_CTRL = 'Controls'


def build_white_mask(bgr_img, h_low, s_low, v_low, h_high, s_high, v_high):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask


def line_centroid(mask, min_area):
    M = cv2.moments(mask)
    area = M['m00']
    if area < min_area:
        return None, area
    cx = int(M['m10'] / area)
    return cx, area


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f'ERROR: Cannot open camera index {CAMERA_INDEX}')
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow(WIN_CAM, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 500, 300)

    # Default values matching white_line_follower.py params
    cv2.createTrackbar('H low',   WIN_CTRL,   0, 180, lambda x: None)
    cv2.createTrackbar('S low',   WIN_CTRL,   0, 255, lambda x: None)
    cv2.createTrackbar('V low',   WIN_CTRL, 180, 255, lambda x: None)
    cv2.createTrackbar('H high',  WIN_CTRL, 180, 180, lambda x: None)
    cv2.createTrackbar('S high',  WIN_CTRL,  70, 255, lambda x: None)
    cv2.createTrackbar('V high',  WIN_CTRL, 255, 255, lambda x: None)
    cv2.createTrackbar('ROI %',   WIN_CTRL,  60, 100, lambda x: None)  # roi_y_start * 100
    cv2.createTrackbar('Min area (x100)', WIN_CTRL, 50, 200, lambda x: None)  # min_area / 100

    frame_count = 0
    t0 = time.time()

    print('Camera test running. Press q to quit, s to save snapshot.')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('WARNING: Failed to grab frame')
            continue

        h, w = frame.shape[:2]

        # Read trackbar values
        h_low  = cv2.getTrackbarPos('H low',   WIN_CTRL)
        s_low  = cv2.getTrackbarPos('S low',   WIN_CTRL)
        v_low  = cv2.getTrackbarPos('V low',   WIN_CTRL)
        h_high = cv2.getTrackbarPos('H high',  WIN_CTRL)
        s_high = cv2.getTrackbarPos('S high',  WIN_CTRL)
        v_high = cv2.getTrackbarPos('V high',  WIN_CTRL)
        roi_pct  = cv2.getTrackbarPos('ROI %', WIN_CTRL)
        min_area = cv2.getTrackbarPos('Min area (x100)', WIN_CTRL) * 100

        y0 = int(h * roi_pct / 100)
        roi = frame[y0:h, 0:w]

        mask = build_white_mask(roi, h_low, s_low, v_low, h_high, s_high, v_high)
        cx, area = line_centroid(mask, min_area)

        # --- Overlay on camera frame ---
        vis = frame.copy()
        # ROI rectangle
        cv2.rectangle(vis, (0, y0), (w - 1, h - 1), (0, 255, 0), 2)
        cv2.putText(vis, f'ROI start: y={y0} ({roi_pct}%)', (5, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Centroid dot
        if cx is not None:
            cy_full = y0 + (h - y0) // 2
            cv2.circle(vis, (cx, cy_full), 10, (0, 0, 255), -1)
            cv2.line(vis, (w // 2, y0), (w // 2, h), (255, 0, 0), 1)  # centre line
            error = cx - w // 2
            status = f'LINE  cx={cx}  error={error:+d}  area={int(area)}'
            color = (0, 255, 0)
        else:
            status = f'NO LINE  area={int(area)}  (min={min_area})'
            color = (0, 0, 255)

        cv2.putText(vis, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # FPS counter
        frame_count += 1
        elapsed = time.time() - t0
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(vis, f'FPS: {fps:.1f}', (w - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

        # Expand mask to 3-ch for display
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        cv2.imshow(WIN_CAM, vis)
        cv2.imshow(WIN_MASK, mask_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print('Quit.')
            break
        elif key == ord('s'):
            ts = int(time.time())
            cv2.imwrite(f'/tmp/frame_{ts}.jpg', frame)
            cv2.imwrite(f'/tmp/mask_{ts}.jpg', mask)
            print(f'Saved /tmp/frame_{ts}.jpg and /tmp/mask_{ts}.jpg')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
