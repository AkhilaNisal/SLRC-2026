#!/usr/bin/env python3
"""
Shared white-line detection utilities used across all task nodes.

Having a single source of truth means HSV tuning, noise-removal kernel
sizes, and the P-controller formula only need to be changed in one place.
"""

import cv2
import numpy as np


def build_white_mask(bgr_img, h_low, s_low, v_low, h_high, s_high, v_high):
    """
    Return a binary mask of white pixels in the given BGR image.

    White is defined in HSV as:
      - H: h_low  .. h_high  (0-180; typically full range 0-180)
      - S: s_low  .. s_high  (0-255; low saturation = near-white/grey)
      - V: v_low  .. v_high  (0-255; high value = bright)

    Gaussian blur + morphological opening remove salt-and-pepper noise and
    small spurious blobs.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask


def line_centroid(mask, min_area):
    """
    Compute the horizontal centroid of the white region in *mask*.

    Returns:
        (cx, area)  where cx is the x-pixel centroid, or
        (None, area) if the white blob is smaller than min_area.
    """
    M = cv2.moments(mask)
    area = M["m00"]
    if area < min_area:
        return None, area
    cx = int(M["m10"] / area)
    return cx, area


def steering_command(cx, frame_width, kp, max_angular,
                     linear_speed, search_linear, search_angular):
    """
    Proportional steering controller.

    Args:
        cx:            Horizontal centroid of the line (None = line lost).
        frame_width:   Width of the ROI/frame in pixels.
        kp:            Proportional gain (positive).
        max_angular:   Maximum allowable angular velocity (rad/s).
        linear_speed:  Forward speed when line is detected (m/s).
        search_linear: Forward speed when line is lost (m/s).
        search_angular: Turn rate when line is lost (rad/s).

    Returns:
        (linear_x, angular_z) as floats.

    Convention:
        error > 0  -> line is to the right  -> angular_z < 0  -> turn right
        error < 0  -> line is to the left   -> angular_z > 0  -> turn left
    """
    if cx is None:
        return search_linear, search_angular
    error = float(cx - frame_width // 2)
    ang = -kp * error
    ang = max(-max_angular, min(max_angular, ang))
    return linear_speed, ang
