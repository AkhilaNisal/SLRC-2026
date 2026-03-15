#!/usr/bin/env python3
"""
Red Box Approach Controller with dynamic path update

States:
- SEARCH_RED
- TURN_TO_SIDE
- FOLLOW_PATH
- DONE
- FAILED

Behavior:
1. Search and lock red box
2. Rotate until the red box moves to the opposite side of the frame
3. Enter FOLLOW_PATH
4. In FOLLOW_PATH:
   - keep target locked
   - dynamically update the path every frame
   - follow path continuously
   - never rotate in place again
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class RedBoxDynamicPathFollower(Node):
    def __init__(self):
        super().__init__('red_box_perpendicular_seeker')

        # --------------------------------------------------
        # Topics
        # --------------------------------------------------
        self.declare_parameter('raw_image_topic', '/camera/image/image_color')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        # --------------------------------------------------
        # Red detection
        # --------------------------------------------------
        self.declare_parameter('h1_low', 0)
        self.declare_parameter('h1_high', 10)
        self.declare_parameter('h2_low', 170)
        self.declare_parameter('h2_high', 180)
        self.declare_parameter('red_s_low', 120)
        self.declare_parameter('red_v_low', 70)
        self.declare_parameter('min_red_area', 150)

        # --------------------------------------------------
        # Search / lock
        # --------------------------------------------------
        self.declare_parameter('lock_tolerance_px', 200.0)
        self.declare_parameter('search_turn_speed', 0.20)
        self.declare_parameter('lost_target_max_frames', 20)

        # --------------------------------------------------
        # Turn-to-side phase
        # --------------------------------------------------
        self.declare_parameter('rotation_speed', 0.30)
        self.declare_parameter('rotation_deadzone_px', 25.0)
        self.declare_parameter('rotation_timeout_sec', 25.0)

        # --------------------------------------------------
        # Path generation
        # --------------------------------------------------
        self.declare_parameter('bezier_num_points', 200)
        self.declare_parameter('bezier_start_tangent_ratio', 0.20)
        self.declare_parameter('bezier_end_tangent_ratio', 0.1)
        self.declare_parameter('bezier_lift_ratio', 0.58)

        # --------------------------------------------------
        # Path following
        # --------------------------------------------------
        self.declare_parameter('lookahead_distance_px', 150.0)
        self.declare_parameter('near_end_lookahead_px', 55.0)

        self.declare_parameter('k_heading', 0.15)
        self.declare_parameter('k_frame_center_x', 0.15)

        self.declare_parameter('max_linear_speed', 0.14)
        self.declare_parameter('slow_linear_speed', 0.08)
        self.declare_parameter('creep_linear_speed', 0.05)
        self.declare_parameter('max_angular_speed', 0.22)

        self.declare_parameter('heading_slow_threshold_deg', 20.0)
        self.declare_parameter('heading_large_threshold_deg', 40.0)

        # --------------------------------------------------
        # Arrival
        # --------------------------------------------------
        self.declare_parameter('stop_distance_px', 40.0)
        self.declare_parameter('stop_box_width_ratio', 0.22)
        self.declare_parameter('stop_area', 9500.0)

        # --------------------------------------------------
        # Robot image anchor
        # --------------------------------------------------
        self.declare_parameter('robot_anchor_y_offset', 20)

        # --------------------------------------------------
        # Timing
        # --------------------------------------------------
        self.declare_parameter('control_rate_hz', 20.0)

        # --------------------------------------------------
        # Read parameters
        # --------------------------------------------------
        self.raw_image_topic = self.get_parameter('raw_image_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.h1_low = int(self.get_parameter('h1_low').value)
        self.h1_high = int(self.get_parameter('h1_high').value)
        self.h2_low = int(self.get_parameter('h2_low').value)
        self.h2_high = int(self.get_parameter('h2_high').value)
        self.red_s_low = int(self.get_parameter('red_s_low').value)
        self.red_v_low = int(self.get_parameter('red_v_low').value)
        self.min_red_area = float(self.get_parameter('min_red_area').value)

        self.lock_tolerance_px = float(self.get_parameter('lock_tolerance_px').value)
        self.search_turn_speed = float(self.get_parameter('search_turn_speed').value)
        self.lost_target_max_frames = int(self.get_parameter('lost_target_max_frames').value)

        self.rotation_speed = float(self.get_parameter('rotation_speed').value)
        self.rotation_deadzone_px = float(self.get_parameter('rotation_deadzone_px').value)
        self.rotation_timeout_sec = float(self.get_parameter('rotation_timeout_sec').value)

        self.bezier_num_points = int(self.get_parameter('bezier_num_points').value)
        self.bezier_start_tangent_ratio = float(self.get_parameter('bezier_start_tangent_ratio').value)
        self.bezier_end_tangent_ratio = float(self.get_parameter('bezier_end_tangent_ratio').value)
        self.bezier_lift_ratio = float(self.get_parameter('bezier_lift_ratio').value)

        self.lookahead_distance_px = float(self.get_parameter('lookahead_distance_px').value)
        self.near_end_lookahead_px = float(self.get_parameter('near_end_lookahead_px').value)

        self.k_heading = float(self.get_parameter('k_heading').value)
        self.k_frame_center_x = float(self.get_parameter('k_frame_center_x').value)

        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.slow_linear_speed = float(self.get_parameter('slow_linear_speed').value)
        self.creep_linear_speed = float(self.get_parameter('creep_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)

        self.heading_slow_threshold_deg = float(self.get_parameter('heading_slow_threshold_deg').value)
        self.heading_large_threshold_deg = float(self.get_parameter('heading_large_threshold_deg').value)

        self.stop_distance_px = float(self.get_parameter('stop_distance_px').value)
        self.stop_box_width_ratio = float(self.get_parameter('stop_box_width_ratio').value)
        self.stop_area = float(self.get_parameter('stop_area').value)

        self.robot_anchor_y_offset = int(self.get_parameter('robot_anchor_y_offset').value)

        self.control_rate_hz = float(self.get_parameter('control_rate_hz').value)

        # --------------------------------------------------
        # ROS
        # --------------------------------------------------
        self.bridge = CvBridge()
        self.raw_frame = None

        self.raw_sub = self.create_subscription(Image, self.raw_image_topic, self.raw_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.timer = self.create_timer(1.0 / self.control_rate_hz, self.process_loop)

        # --------------------------------------------------
        # State / action data
        # --------------------------------------------------
        self.state = "SEARCH_RED"

        self.locked_red_box = None
        self.last_visible_red_box = None
        self.lost_target_frames = 0

        self.initial_red_offset = None
        self.initial_red_side = None
        self.target_offset = None
        self.rotation_start_time = None

        self.curved_path = None
        self.path_control_points = None
        self.nearest_path_index = 0
        self.nearest_path_point = None
        self.lookahead_point = None

        self.heading_error = 0.0
        self.frame_center_error_x = 0.0
        self.last_cmd_vx = 0.0
        self.last_cmd_wz = 0.0

        cv2.namedWindow("red_box_dynamic_path_follower", cv2.WINDOW_NORMAL)

        self.get_logger().info("RedBoxDynamicPathFollower started.")
        self.get_logger().info("States: SEARCH_RED -> TURN_TO_SIDE -> FOLLOW_PATH -> DONE")

    # ==================================================
    # ROS callbacks / helpers
    # ==================================================
    def raw_cb(self, msg):
        self.raw_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def publish_cmd(self, vx, wz):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)
        self.last_cmd_vx = float(vx)
        self.last_cmd_wz = float(wz)

    def stop(self):
        self.publish_cmd(0.0, 0.0)

    def clamp(self, val, low, high):
        return max(low, min(high, val))

    def reset_all(self):
        self.state = "SEARCH_RED"
        self.locked_red_box = None
        self.last_visible_red_box = None
        self.lost_target_frames = 0

        self.initial_red_offset = None
        self.initial_red_side = None
        self.target_offset = None
        self.rotation_start_time = None

        self.curved_path = None
        self.path_control_points = None
        self.nearest_path_index = 0
        self.nearest_path_point = None
        self.lookahead_point = None

        self.heading_error = 0.0
        self.frame_center_error_x = 0.0
        self.stop()

    # ==================================================
    # Detection
    # ==================================================
    def build_red_mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        lower1 = np.array([self.h1_low, self.red_s_low, self.red_v_low], dtype=np.uint8)
        upper1 = np.array([self.h1_high, 255, 255], dtype=np.uint8)
        lower2 = np.array([self.h2_low, self.red_s_low, self.red_v_low], dtype=np.uint8)
        upper2 = np.array([self.h2_high, 255, 255], dtype=np.uint8)

        m1 = cv2.inRange(hsv, lower1, upper1)
        m2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(m1, m2)

        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return mask

    def extract_red_candidates(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_red_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            if M["m00"] <= 1e-6:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            candidates.append((x, y, w, h, cx, cy, area))

        return candidates

    def detect_largest_red_box(self, mask):
        candidates = self.extract_red_candidates(mask)
        if not candidates:
            return None
        return max(candidates, key=lambda b: b[6])

    def detect_locked_or_best_red_box(self, mask, prev_box=None, tolerance_px=200.0):
        candidates = self.extract_red_candidates(mask)
        if not candidates:
            return None

        if prev_box is None:
            return max(candidates, key=lambda b: b[6])

        _, _, _, _, prev_cx, prev_cy, _ = prev_box

        near_best = None
        near_best_dist = float('inf')
        for box in candidates:
            _, _, _, _, cx, cy, _ = box
            dist = np.hypot(cx - prev_cx, cy - prev_cy)
            if dist <= tolerance_px and dist < near_best_dist:
                near_best_dist = dist
                near_best = box

        if near_best is not None:
            return near_best

        return max(candidates, key=lambda b: b[6])

    def get_box_offset_and_side(self, box_cx, frame_width):
        center_x = frame_width / 2.0
        offset = float(box_cx - center_x)
        side = 'left' if offset < 0.0 else 'right'
        return offset, side

    # ==================================================
    # Geometry / path
    # ==================================================
    def norm2(self, v):
        return float(np.hypot(v[0], v[1]))

    def unit(self, v):
        n = self.norm2(v)
        if n < 1e-6:
            return np.array([0.0, -1.0], dtype=np.float64)
        return np.array([v[0] / n, v[1] / n], dtype=np.float64)

    def cubic_bezier(self, p0, p1, p2, p3, n):
        pts = []
        for i in range(n):
            t = i / float(n - 1) if n > 1 else 0.0
            u = 1.0 - t
            b = (
                (u ** 3) * p0 +
                3.0 * (u ** 2) * t * p1 +
                3.0 * u * (t ** 2) * p2 +
                (t ** 3) * p3
            )
            pts.append(b)
        return np.array(pts, dtype=np.float64)

    def generate_curved_path(self, robot_start, red_box_bottom_center):
        p0 = np.array(robot_start, dtype=np.float64)
        p3 = np.array(red_box_bottom_center, dtype=np.float64)

        dist = max(40.0, self.norm2(p3 - p0))

        start_dir = np.array([0.0, -1.0], dtype=np.float64)
        end_dir = np.array([0.0, 1.0], dtype=np.float64)

        start_len = self.bezier_start_tangent_ratio * dist
        end_len = self.bezier_end_tangent_ratio * dist

        p1 = p0 + start_len * start_dir
        p2 = p3 - end_len * end_dir

        direction = self.unit(p3 - p0)
        perp = np.array([-direction[1], direction[0]], dtype=np.float64)
        lift = dist * self.bezier_lift_ratio

        p1 = p1 + perp * lift * 0.3
        p2 = p2 + perp * lift * 0.3

        curve_pts = self.cubic_bezier(p0, p1, p2, p3, self.bezier_num_points)
        return curve_pts, p1, p2

    def cumulative_path_lengths(self, pts):
        if pts is None or len(pts) < 2:
            return np.array([0.0], dtype=np.float64)
        ds = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        return np.concatenate(([0.0], np.cumsum(ds)))

    def find_nearest_point_on_path(self, robot_pos):
        if self.curved_path is None or len(self.curved_path) == 0:
            return None, 0, 0.0

        robot_arr = np.array(robot_pos, dtype=np.float64)
        dists = np.linalg.norm(self.curved_path - robot_arr, axis=1)
        idx = int(np.argmin(dists))
        return tuple(self.curved_path[idx].astype(int)), idx, float(dists[idx])

    def find_lookahead_point(self, nearest_idx, lookahead_dist):
        if self.curved_path is None or len(self.curved_path) == 0:
            return None, nearest_idx

        s = self.cumulative_path_lengths(self.curved_path)
        target_s = s[nearest_idx] + lookahead_dist

        idx = nearest_idx
        while idx + 1 < len(self.curved_path) and s[idx] < target_s:
            idx += 1

        return tuple(self.curved_path[idx].astype(int)), idx

    def distance_remaining_on_path(self, nearest_idx):
        if self.curved_path is None or len(self.curved_path) < 2:
            return 0.0
        s = self.cumulative_path_lengths(self.curved_path)
        if nearest_idx < 0 or nearest_idx >= len(s):
            return 0.0
        return float(s[-1] - s[nearest_idx])

    def compute_heading_error(self, robot_pos, target_pos):
        dx = float(target_pos[0] - robot_pos[0])
        dy = float(target_pos[1] - robot_pos[1])
        return float(np.arctan2(dx, -dy))

    def compute_frame_center_error(self, red_box_pos, frame_width):
        frame_center_x = frame_width / 2.0
        return float(red_box_pos[0] - frame_center_x)

    def compute_path_following_control(self, frame_shape, red_box, robot_pos, lookahead_point, nearest_idx):
        h, w = frame_shape[:2]

        remaining = self.distance_remaining_on_path(nearest_idx)
        if remaining <= self.stop_distance_px:
            return 0.0, 0.0, "DONE"

        width_ratio = 0.0
        center_term = 0.0

        if red_box is not None:
            x, y, bw, bh, cx, cy, area = red_box
            width_ratio = bw / float(w)

            if area >= self.stop_area or width_ratio >= self.stop_box_width_ratio:
                return 0.0, 0.0, "DONE"

            self.frame_center_error_x = self.compute_frame_center_error((cx, cy), w)
            center_term = self.k_frame_center_x * np.radians(self.frame_center_error_x / 10.0)
        else:
            self.frame_center_error_x = 0.0
            center_term = 0.0

        if lookahead_point is None:
            return self.creep_linear_speed, 0.0, "BLIND_FORWARD"

        self.heading_error = self.compute_heading_error(robot_pos, lookahead_point)
        heading_deg = abs(np.degrees(self.heading_error))

        heading_term = self.k_heading * self.heading_error

        # reversed sign for correct follow direction
        wz = -(heading_term + center_term)
        wz = self.clamp(wz, -self.max_angular_speed, self.max_angular_speed)

        if heading_deg > self.heading_large_threshold_deg:
            vx = self.creep_linear_speed
            state = "FOLLOW_LARGE_ERR"
        elif heading_deg > self.heading_slow_threshold_deg:
            vx = self.slow_linear_speed
            state = "FOLLOW_MEDIUM_ERR"
        else:
            vx = self.max_linear_speed
            state = "FOLLOW_PATH"

        return vx, wz, state

    # ==================================================
    # State handlers
    # ==================================================
    def handle_search_red(self, raw, red_mask):
        red_box = self.detect_largest_red_box(red_mask)
        if red_box is None:
            self.publish_cmd(0.0, self.search_turn_speed)
            return None

        self.locked_red_box = red_box
        self.last_visible_red_box = red_box
        self.lost_target_frames = 0

        _, _, _, _, cx, _, _ = red_box
        offset, side = self.get_box_offset_and_side(cx, raw.shape[1])

        self.initial_red_offset = offset
        self.initial_red_side = side
        self.target_offset = -offset
        self.rotation_start_time = self.get_clock().now()

        self.state = "TURN_TO_SIDE"
        self.get_logger().info(
            f"Locked red box on {side.upper()} at offset {offset:.1f}px -> TURN_TO_SIDE"
        )
        self.stop()
        return red_box

    def handle_turn_to_side(self, raw, red_mask):
        red_box = self.detect_locked_or_best_red_box(
            red_mask, self.locked_red_box, tolerance_px=self.lock_tolerance_px
        )

        if red_box is None:
            self.lost_target_frames += 1
            if self.lost_target_frames > self.lost_target_max_frames:
                self.get_logger().warn("Lost target in TURN_TO_SIDE -> SEARCH_RED")
                self.reset_all()
            else:
                self.publish_cmd(0.0, self.search_turn_speed)
            return self.locked_red_box

        self.locked_red_box = red_box
        self.last_visible_red_box = red_box
        self.lost_target_frames = 0

        _, _, _, _, cx, _, _ = red_box
        current_offset, current_side = self.get_box_offset_and_side(cx, raw.shape[1])

        elapsed = (self.get_clock().now() - self.rotation_start_time).nanoseconds / 1e9
        if elapsed > self.rotation_timeout_sec:
            self.get_logger().warn("TURN_TO_SIDE timeout -> SEARCH_RED")
            self.reset_all()
            return red_box

        reached = False
        if self.target_offset is not None:
            if self.target_offset > 0:
                reached = current_offset > (self.target_offset - self.rotation_deadzone_px)
            else:
                reached = current_offset < (self.target_offset + self.rotation_deadzone_px)

        if reached:
            self.state = "FOLLOW_PATH"
            self.stop()
            self.get_logger().info(
                f"Reached opposite side ({current_side.upper()}, {current_offset:.1f}px) -> FOLLOW_PATH"
            )
            return red_box

        wz = self.rotation_speed if current_offset < self.target_offset else -self.rotation_speed
        self.publish_cmd(0.0, wz)
        return red_box

    def handle_follow_path(self, raw, red_mask):
        # Keep target locked and dynamically update path each frame
        red_box = self.detect_locked_or_best_red_box(
            red_mask, self.locked_red_box, tolerance_px=self.lock_tolerance_px + 40.0
        )

        if red_box is not None:
            self.locked_red_box = red_box
            self.last_visible_red_box = red_box
            self.lost_target_frames = 0
        else:
            self.lost_target_frames += 1
            red_box = self.last_visible_red_box

            if red_box is None and self.lost_target_frames > self.lost_target_max_frames:
                self.get_logger().warn("Lost target in FOLLOW_PATH -> FAILED")
                self.state = "FAILED"
                self.stop()
                return None

        h, w = raw.shape[:2]
        robot_start = (w // 2, h - self.robot_anchor_y_offset)

        # Dynamic target update
        if red_box is not None:
            x, y, bw, bh, cx, cy, area = red_box
            red_bottom_center = (x + bw // 2, y + bh)

            self.curved_path, p1, p2 = self.generate_curved_path(robot_start, red_bottom_center)
            self.path_control_points = (
                robot_start,
                tuple(np.array(p1, dtype=int)),
                tuple(np.array(p2, dtype=int)),
                red_bottom_center
            )

        self.nearest_path_point, self.nearest_path_index, _ = self.find_nearest_point_on_path(robot_start)

        remaining = self.distance_remaining_on_path(self.nearest_path_index)
        lookahead_dist = self.near_end_lookahead_px if remaining < 120.0 else self.lookahead_distance_px
        self.lookahead_point, _ = self.find_lookahead_point(self.nearest_path_index, lookahead_dist)

        vx, wz, follow_state = self.compute_path_following_control(
            raw.shape, red_box, robot_start, self.lookahead_point, self.nearest_path_index
        )

        if follow_state == "DONE":
            self.state = "DONE"
            self.stop()
            self.get_logger().info("Reached red box -> DONE")
        else:
            self.publish_cmd(vx, wz)

        return red_box

    def handle_done(self, raw, red_mask):
        red_box = self.detect_locked_or_best_red_box(
            red_mask, self.locked_red_box, tolerance_px=self.lock_tolerance_px + 50.0
        )
        if red_box is not None:
            self.locked_red_box = red_box
            self.last_visible_red_box = red_box
        self.stop()
        return self.locked_red_box

    def handle_failed(self, raw, red_mask):
        self.stop()
        return self.locked_red_box

    # ==================================================
    # Main loop
    # ==================================================
    def process_loop(self):
        if self.raw_frame is None:
            self.stop()
            return

        raw = self.raw_frame.copy()
        h, w = raw.shape[:2]
        robot_start = (w // 2, h - self.robot_anchor_y_offset)

        red_mask = self.build_red_mask(raw)
        red_box = None

        if self.state == "SEARCH_RED":
            red_box = self.handle_search_red(raw, red_mask)
        elif self.state == "TURN_TO_SIDE":
            red_box = self.handle_turn_to_side(raw, red_mask)
        elif self.state == "FOLLOW_PATH":
            red_box = self.handle_follow_path(raw, red_mask)
        elif self.state == "DONE":
            red_box = self.handle_done(raw, red_mask)
        elif self.state == "FAILED":
            red_box = self.handle_failed(raw, red_mask)
        else:
            self.get_logger().warn(f"Unknown state: {self.state}")
            self.reset_all()

        self.draw_debug(raw, red_box, robot_start)
        cv2.imshow("red_box_dynamic_path_follower", raw)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.stop()
            rclpy.shutdown()

    # ==================================================
    # Debug drawing
    # ==================================================
    def draw_debug(self, img, red_box, robot_start):
        h, w = img.shape[:2]
        center_x = w // 2

        cv2.line(img, (center_x, 0), (center_x, h - 1), (255, 0, 255), 1)
        cv2.circle(img, robot_start, 8, (0, 255, 0), -1)
        cv2.putText(img, "ROBOT", (robot_start[0] + 8, robot_start[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self.state == "TURN_TO_SIDE" and self.target_offset is not None:
            target_x = int(center_x + self.target_offset)
            dz = int(self.rotation_deadzone_px)
            cv2.rectangle(img, (target_x - dz, 0), (target_x + dz, h - 1), (100, 200, 100), 2)
            cv2.putText(img, "TARGET ZONE", (max(10, target_x - 70), 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 2)

        if red_box is not None:
            x, y, bw, bh, cx, cy, area = red_box
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cv2.circle(img, (cx, cy), 6, (0, 255, 255), -1)
            bc = (x + bw // 2, y + bh)
            cv2.circle(img, bc, 6, (255, 255, 0), -1)

            offset, side = self.get_box_offset_and_side(cx, w)
            cv2.putText(img, f"offset={offset:.1f}px side={side.upper()}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(img, f"area={int(area)} width_ratio={bw / float(w):.3f}",
                        (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 210, 255), 2)

        if self.curved_path is not None and len(self.curved_path) >= 2:
            for i in range(len(self.curved_path) - 1):
                a = tuple(self.curved_path[i].astype(int))
                b = tuple(self.curved_path[i + 1].astype(int))
                color = (120, 120, 120) if i < self.nearest_path_index else (0, 255, 0)
                cv2.line(img, a, b, color, 2)

        if self.path_control_points is not None:
            p0, p1, p2, p3 = self.path_control_points
            cv2.circle(img, p0, 7, (255, 0, 0), -1)
            cv2.circle(img, p1, 5, (0, 180, 0), -1)
            cv2.circle(img, p2, 5, (0, 180, 255), -1)
            cv2.circle(img, p3, 7, (255, 0, 255), -1)

        if self.nearest_path_point is not None:
            cv2.circle(img, self.nearest_path_point, 6, (180, 0, 180), -1)

        if self.lookahead_point is not None:
            cv2.circle(img, self.lookahead_point, 8, (0, 165, 255), -1)
            cv2.putText(img, "LOOKAHEAD", (self.lookahead_point[0] + 6, self.lookahead_point[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            cv2.line(img, robot_start, self.lookahead_point, (0, 165, 255), 1)

        state_color = {
            "SEARCH_RED": (0, 165, 255),
            "TURN_TO_SIDE": (0, 255, 255),
            "FOLLOW_PATH": (0, 255, 0),
            "DONE": (0, 255, 0),
            "FAILED": (0, 0, 255),
        }.get(self.state, (200, 200, 200))

        cv2.putText(img, f"STATE: {self.state}", (10, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_color, 2)
        cv2.putText(img, f"vx={self.last_cmd_vx:.3f} wz={self.last_cmd_wz:.3f}",
                    (10, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"heading_err_deg={np.degrees(self.heading_error):.2f}",
                    (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    def destroy_node(self):
        try:
            self.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = RedBoxDynamicPathFollower()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()