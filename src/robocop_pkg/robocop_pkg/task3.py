#!/usr/bin/env python3

# starting from:     translation 1.2 1 0
#                      rotation 0 0 1 1.57

import cv2
import numpy as np

from robocop_pkg.line_detection_utils import build_white_mask
import rclpy

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Range
from std_msgs.msg import String

from robot_arm_interfaces.action import PickBox


class Task3Node(Node):
    def __init__(self):
        super().__init__('task3')

        # =========================
        # Topics
        # =========================
        self.declare_parameter('image_topic', '/camera/image/image_color')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('front_range_topic', '/robocop/ds_front')
        self.declare_parameter('task3_status_topic', '/task3/status')

        # =========================
        # Arm action
        # =========================
        self.declare_parameter('arm_action_name', '/pick_box')
        self.declare_parameter('trigger_restore_after_stop', True)

        # =========================
        # Motion
        # Do not change these calibration values
        # =========================
        self.declare_parameter('linear_speed', 0.2)
        self.declare_parameter('search_linear', 0.05)
        self.declare_parameter('search_angular', 0.35)

        self.declare_parameter('kp_line', 0.004)
        self.declare_parameter('max_angular_line', 1.2)

        self.declare_parameter('turn_left_angular_speed', 0.8)
        self.declare_parameter('turn_left_90_time', 2.85)
        self.declare_parameter('post_turn_wait_time', 0.8)

        self.declare_parameter('junction_turn_speed', 0.8)
        self.declare_parameter('junction_turn_90_time', 2.85)
        self.declare_parameter('junction_confirm_frames', 4)

        self.declare_parameter('pre_turn_distance', 0.36)
        self.declare_parameter('pre_turn_speed', 0.10)

        # =========================
        # White line detection
        # =========================
        self.declare_parameter('follow_roi_y_start', 0.55)
        self.declare_parameter('follow_roi_x_min_ratio', 0.15)
        self.declare_parameter('follow_roi_x_max_ratio', 0.85)
        self.declare_parameter('min_area', 5000)
        self.declare_parameter('startup_line_detect_min_area', 2500)

        self.declare_parameter('h_low', 0)
        self.declare_parameter('s_low', 0)
        self.declare_parameter('v_low', 180)
        self.declare_parameter('h_high', 180)
        self.declare_parameter('s_high', 70)
        self.declare_parameter('v_high', 255)

        # =========================
        # Branch detection
        # =========================
        self.declare_parameter('branch_side', 'LEFT')
        self.declare_parameter('side_roi_x_ratio', 0.34)
        self.declare_parameter('bottom_strip_height_ratio', 0.15)
        self.declare_parameter('bottom_branch_min_area', 2000)
        self.declare_parameter('target_branch_index', 2)
        self.declare_parameter('require_main_line_for_branch', True)

        # =========================
        # Marker detection
        # =========================
        self.declare_parameter('detect_marker_after_branch', True)
        self.declare_parameter('marker_confirm_frames', 4)

        # Green
        self.declare_parameter('green_h_low', 35)
        self.declare_parameter('green_h_high', 95)
        self.declare_parameter('green_s_low', 90)
        self.declare_parameter('green_v_low', 60)

        # Blue
        self.declare_parameter('blue_h_low', 95)
        self.declare_parameter('blue_h_high', 135)
        self.declare_parameter('blue_s_low', 100)
        self.declare_parameter('blue_v_low', 50)

        # Shared blob checks
        self.declare_parameter('marker_min_area', 120)
        self.declare_parameter('marker_max_area', 12000)
        self.declare_parameter('marker_min_circularity', 0.55)

        # Green pair checks
        self.declare_parameter('green_pair_max_dx', 70)
        self.declare_parameter('green_pair_min_dy', 40)

        # Blue pair checks
        self.declare_parameter('blue_pair_max_dy', 60)
        self.declare_parameter('blue_pair_min_dx', 60)

        # Blue ROI to avoid sky
        self.declare_parameter('blue_roi_y_min_ratio', 0.18)
        self.declare_parameter('blue_roi_y_max_ratio', 0.88)
        self.declare_parameter('blue_roi_x_min_ratio', 0.08)
        self.declare_parameter('blue_roi_x_max_ratio', 0.92)

        # Final alignment
        self.declare_parameter('marker_kp_align', 0.005)
        self.declare_parameter('marker_max_angular', 0.9)
        self.declare_parameter('marker_align_tolerance_px', 25)
        self.declare_parameter('marker_approach_speed', 0.08)
        self.declare_parameter('marker_search_angular', 0.25)

        # Front stop
        self.declare_parameter('front_stop_distance', 0.24)
        self.declare_parameter('front_slow_distance', 0.35)

        # =========================
        # Read params
        # =========================
        self.image_topic = self.get_parameter('image_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.front_range_topic = self.get_parameter('front_range_topic').value
        self.task3_status_topic = self.get_parameter('task3_status_topic').value

        self.arm_action_name = str(self.get_parameter('arm_action_name').value)
        self.trigger_restore_after_stop = bool(self.get_parameter('trigger_restore_after_stop').value)

        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.search_linear = float(self.get_parameter('search_linear').value)
        self.search_angular = float(self.get_parameter('search_angular').value)

        self.kp_line = float(self.get_parameter('kp_line').value)
        self.max_angular_line = float(self.get_parameter('max_angular_line').value)

        self.turn_left_angular_speed = float(self.get_parameter('turn_left_angular_speed').value)
        self.turn_left_90_time = float(self.get_parameter('turn_left_90_time').value)
        self.post_turn_wait_time = float(self.get_parameter('post_turn_wait_time').value)

        self.junction_turn_speed = float(self.get_parameter('junction_turn_speed').value)
        self.junction_turn_90_time = float(self.get_parameter('junction_turn_90_time').value)
        self.junction_confirm_frames = int(self.get_parameter('junction_confirm_frames').value)

        self.pre_turn_distance = float(self.get_parameter('pre_turn_distance').value)
        self.pre_turn_speed = float(self.get_parameter('pre_turn_speed').value)
        self.pre_turn_time = (
            self.pre_turn_distance / self.pre_turn_speed
            if self.pre_turn_speed > 1e-6 else 0.0
        )

        self.follow_roi_y_start = float(self.get_parameter('follow_roi_y_start').value)
        self.follow_roi_x_min_ratio = float(self.get_parameter('follow_roi_x_min_ratio').value)
        self.follow_roi_x_max_ratio = float(self.get_parameter('follow_roi_x_max_ratio').value)
        self.min_area = int(self.get_parameter('min_area').value)
        self.startup_line_detect_min_area = int(self.get_parameter('startup_line_detect_min_area').value)

        self.h_low = int(self.get_parameter('h_low').value)
        self.s_low = int(self.get_parameter('s_low').value)
        self.v_low = int(self.get_parameter('v_low').value)
        self.h_high = int(self.get_parameter('h_high').value)
        self.s_high = int(self.get_parameter('s_high').value)
        self.v_high = int(self.get_parameter('v_high').value)

        self.branch_side = str(self.get_parameter('branch_side').value).strip().upper()
        self.side_roi_x_ratio = float(self.get_parameter('side_roi_x_ratio').value)
        self.bottom_strip_height_ratio = float(self.get_parameter('bottom_strip_height_ratio').value)
        self.bottom_branch_min_area = int(self.get_parameter('bottom_branch_min_area').value)
        self.target_branch_index = int(self.get_parameter('target_branch_index').value)
        self.require_main_line_for_branch = bool(self.get_parameter('require_main_line_for_branch').value)

        self.detect_marker_after_branch = bool(self.get_parameter('detect_marker_after_branch').value)
        self.marker_confirm_frames = int(self.get_parameter('marker_confirm_frames').value)

        self.green_h_low = int(self.get_parameter('green_h_low').value)
        self.green_h_high = int(self.get_parameter('green_h_high').value)
        self.green_s_low = int(self.get_parameter('green_s_low').value)
        self.green_v_low = int(self.get_parameter('green_v_low').value)

        self.blue_h_low = int(self.get_parameter('blue_h_low').value)
        self.blue_h_high = int(self.get_parameter('blue_h_high').value)
        self.blue_s_low = int(self.get_parameter('blue_s_low').value)
        self.blue_v_low = int(self.get_parameter('blue_v_low').value)

        self.marker_min_area = int(self.get_parameter('marker_min_area').value)
        self.marker_max_area = int(self.get_parameter('marker_max_area').value)
        self.marker_min_circularity = float(self.get_parameter('marker_min_circularity').value)

        self.green_pair_max_dx = int(self.get_parameter('green_pair_max_dx').value)
        self.green_pair_min_dy = int(self.get_parameter('green_pair_min_dy').value)

        self.blue_pair_max_dy = int(self.get_parameter('blue_pair_max_dy').value)
        self.blue_pair_min_dx = int(self.get_parameter('blue_pair_min_dx').value)

        self.blue_roi_y_min_ratio = float(self.get_parameter('blue_roi_y_min_ratio').value)
        self.blue_roi_y_max_ratio = float(self.get_parameter('blue_roi_y_max_ratio').value)
        self.blue_roi_x_min_ratio = float(self.get_parameter('blue_roi_x_min_ratio').value)
        self.blue_roi_x_max_ratio = float(self.get_parameter('blue_roi_x_max_ratio').value)

        self.marker_kp_align = float(self.get_parameter('marker_kp_align').value)
        self.marker_max_angular = float(self.get_parameter('marker_max_angular').value)
        self.marker_align_tolerance_px = int(self.get_parameter('marker_align_tolerance_px').value)
        self.marker_approach_speed = float(self.get_parameter('marker_approach_speed').value)
        self.marker_search_angular = float(self.get_parameter('marker_search_angular').value)

        self.front_stop_distance = float(self.get_parameter('front_stop_distance').value)
        self.front_slow_distance = float(self.get_parameter('front_slow_distance').value)

        if self.branch_side not in ['LEFT', 'RIGHT']:
            self.branch_side = 'LEFT'
            self.get_logger().warn("Invalid branch_side. Using LEFT.")

        if self.target_branch_index < 1:
            self.target_branch_index = 1
            self.get_logger().warn("target_branch_index must be >= 1. Using 1.")

        # =========================
        # ROS interfaces
        # =========================
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_cb, qos_profile_sensor_data
        )
        self.front_sub = self.create_subscription(
            Range, self.front_range_topic, self.front_range_cb, qos_profile_sensor_data
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.task_status_pub = self.create_publisher(String, self.task3_status_topic, 10)

        self.arm_client = ActionClient(self, PickBox, self.arm_action_name)

        # =========================
        # States
        # =========================
        self.STATE_INITIAL_TURN = 'INITIAL_TURN'
        self.STATE_INITIAL_WAIT = 'INITIAL_WAIT'
        self.STATE_FIND_LINE_AFTER_INITIAL_TURN = 'FIND_LINE_AFTER_INITIAL_TURN'
        self.STATE_FOLLOW_LINE_TO_BRANCH = 'FOLLOW_LINE_TO_BRANCH'
        self.STATE_MOVE_FORWARD_BEFORE_TURN = 'MOVE_FORWARD_BEFORE_TURN'
        self.STATE_TURN_AT_BRANCH = 'TURN_AT_BRANCH'
        self.STATE_POST_BRANCH_WAIT = 'POST_BRANCH_WAIT'
        self.STATE_FOLLOW_LINE_AFTER_BRANCH = 'FOLLOW_LINE_AFTER_BRANCH'
        self.STATE_ALIGN_MARKER = 'ALIGN_MARKER'
        self.STATE_APPROACH_MARKER = 'APPROACH_MARKER'
        self.STATE_STOPPED = 'STOPPED'
        self.STATE_WAIT_ARM_SERVER = 'WAIT_ARM_SERVER'
        self.STATE_RESTORE_RUNNING = 'RESTORE_RUNNING'
        self.STATE_RESTORE_DONE = 'RESTORE_DONE'
        self.STATE_RESTORE_FAILED = 'RESTORE_FAILED'

        self.state = self.STATE_INITIAL_TURN
        self.turn_start_time = None
        self.wait_start_time = None
        self.pre_turn_start_time = None
        self.initial_turn_started = False

        self.side_branch_counter = 0
        self.branch_count = 0
        self.branch_currently_visible = False
        self.marker_seen_counter = 0

        self.front_range = None
        self.front_range_valid = False

        self.last_marker_detected = False
        self.last_marker_center = None
        self.last_green_centers = []
        self.last_blue_centers = []
        self.last_marker_mode = "NONE"

        self.restore_goal_sent = False
        self.restore_goal_done = False
        self.restore_goal_success = False
        self.restore_result_message = ""
        self.arm_server_wait_started = False

        self.task3_done = False
        self.task3_done_published = False

        # ---- physical robot fix ----
        # When the robot gets very close to the marker, the circles can leave the camera view.
        # In that case the old code started rotating in APPROACH_MARKER.
        # These variables make the approach state "latch" success instead of searching forever.
        self.approach_marker_seen_once = False
        self.approach_marker_lost_counter = 0
        self.approach_marker_lost_limit = 8

        self.frame_count = 0
        self.last_log_time = self.get_clock().now()

        # cv2.namedWindow("task3_camera", cv2.WINDOW_NORMAL)

        self.get_logger().info(
            f"Task 3 started. Branch side: {self.branch_side}. "
            f"Uses center/lower ROI for main line following, bottom side strip for branch counting, "
            f"turns at branch #{self.target_branch_index}, then detects marker and stops with front sensor."
        )

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def publish_task_done(self):
        if self.task3_done_published:
            return
        msg = String()
        msg.data = 'DONE'
        self.task_status_pub.publish(msg)
        self.task3_done_published = True
        self.get_logger().info("Published Task 3 DONE status.")

    def front_range_cb(self, msg: Range):
        value = float(msg.range)
        if np.isfinite(value):
            self.front_range = value
            self.front_range_valid = True
        else:
            self.front_range_valid = False

    def build_green_mask(self, bgr_img):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        lower = np.array([self.green_h_low, self.green_s_low, self.green_v_low], dtype=np.uint8)
        upper = np.array([self.green_h_high, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def build_blue_mask(self, bgr_img):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        lower = np.array([self.blue_h_low, self.blue_s_low, self.blue_v_low], dtype=np.uint8)
        upper = np.array([self.blue_h_high, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def extract_circle_like_centers(self, mask, x_offset=0, y_offset=0):
        centers = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.marker_min_area or area > self.marker_max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1e-6:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.marker_min_circularity:
                continue

            M = cv2.moments(cnt)
            if M["m00"] <= 1e-6:
                continue

            cx = int(M["m10"] / M["m00"]) + x_offset
            cy = int(M["m01"] / M["m00"]) + y_offset

            centers.append({
                "center": (cx, cy),
                "area": area,
                "circularity": circularity
            })

        centers.sort(key=lambda c: c["area"], reverse=True)
        return centers

    def detect_marker(self, frame):
        h, w = frame.shape[:2]

        green_mask = self.build_green_mask(frame)
        green_candidates = self.extract_circle_like_centers(green_mask)

        blue_x0 = int(w * self.blue_roi_x_min_ratio)
        blue_x1 = int(w * self.blue_roi_x_max_ratio)
        blue_y0 = int(h * self.blue_roi_y_min_ratio)
        blue_y1 = int(h * self.blue_roi_y_max_ratio)

        blue_x0 = max(0, min(blue_x0, w - 1))
        blue_x1 = max(blue_x0 + 1, min(blue_x1, w))
        blue_y0 = max(0, min(blue_y0, h - 1))
        blue_y1 = max(blue_y0 + 1, min(blue_y1, h))

        blue_roi = frame[blue_y0:blue_y1, blue_x0:blue_x1]
        blue_mask_roi = self.build_blue_mask(blue_roi)
        blue_mask = np.zeros((h, w), dtype=np.uint8)
        blue_mask[blue_y0:blue_y1, blue_x0:blue_x1] = blue_mask_roi

        blue_candidates = self.extract_circle_like_centers(
            blue_mask_roi,
            x_offset=blue_x0,
            y_offset=blue_y0
        )

        green_valid = False
        blue_valid = False
        marker_center = None
        marker_mode = "NONE"
        green_centers = []
        blue_centers = []

        if len(green_candidates) >= 2:
            green_centers = [c["center"] for c in green_candidates[:2]]
            green_centers = sorted(green_centers, key=lambda p: p[1])

            top_pt = green_centers[0]
            bottom_pt = green_centers[1]
            dx = abs(top_pt[0] - bottom_pt[0])
            dy = abs(top_pt[1] - bottom_pt[1])

            if dx <= self.green_pair_max_dx and dy >= self.green_pair_min_dy:
                green_valid = True

        if len(blue_candidates) >= 2:
            blue_centers = [c["center"] for c in blue_candidates[:2]]
            blue_centers = sorted(blue_centers, key=lambda p: p[0])

            left_pt = blue_centers[0]
            right_pt = blue_centers[1]
            dx = abs(left_pt[0] - right_pt[0])
            dy = abs(left_pt[1] - right_pt[1])

            if dy <= self.blue_pair_max_dy and dx >= self.blue_pair_min_dx:
                blue_valid = True

        if green_valid and blue_valid:
            gx = int((green_centers[0][0] + green_centers[1][0]) / 2.0)
            gy = int((green_centers[0][1] + green_centers[1][1]) / 2.0)
            bx = int((blue_centers[0][0] + blue_centers[1][0]) / 2.0)
            by = int((blue_centers[0][1] + blue_centers[1][1]) / 2.0)

            marker_center = (int((gx + bx) / 2.0), int((gy + by) / 2.0))
            marker_mode = "GREEN+BLUE"

        elif green_valid:
            marker_center = (
                int((green_centers[0][0] + green_centers[1][0]) / 2.0),
                int((green_centers[0][1] + green_centers[1][1]) / 2.0)
            )
            marker_mode = "GREEN"

        elif blue_valid:
            marker_center = (
                int((blue_centers[0][0] + blue_centers[1][0]) / 2.0),
                int((blue_centers[0][1] + blue_centers[1][1]) / 2.0)
            )
            marker_mode = "BLUE"

        marker_found = marker_center is not None

        self.last_marker_detected = marker_found
        self.last_marker_center = marker_center
        self.last_green_centers = green_centers
        self.last_blue_centers = blue_centers
        self.last_marker_mode = marker_mode

        return marker_found, marker_center, marker_mode, green_mask, blue_mask, (blue_x0, blue_y0, blue_x1, blue_y1)

    def follow_line(self, area, moments, frame_center_x, twist, x_offset=0):
        if area > self.min_area:
            cx_local = int(moments["m10"] / area)
            cx = x_offset + cx_local
            error = float(cx - frame_center_x)
            ang = -self.kp_line * error
            ang = self.clamp(ang, -self.max_angular_line, self.max_angular_line)

            twist.linear.x = self.linear_speed
            twist.angular.z = ang
        else:
            twist.linear.x = self.search_linear
            twist.angular.z = self.search_angular

    def align_to_marker(self, marker_center, frame_width, twist, allow_forward=False):
        if marker_center is None:
            twist.linear.x = 0.0
            twist.angular.z = self.marker_search_angular
            return

        error_x = float(marker_center[0] - (frame_width // 2))
        ang = -self.marker_kp_align * error_x
        ang = self.clamp(ang, -self.marker_max_angular, self.marker_max_angular)

        if abs(error_x) <= self.marker_align_tolerance_px:
            twist.linear.x = self.marker_approach_speed if allow_forward else 0.0
            twist.angular.z = ang
        else:
            twist.linear.x = 0.03 if allow_forward else 0.0
            twist.angular.z = ang

    def try_start_restore_action(self):
        if self.restore_goal_sent:
            return

        if not self.arm_client.server_is_ready():
            if not self.arm_server_wait_started:
                self.get_logger().info(f"Waiting for arm action server: {self.arm_action_name}")
                self.arm_server_wait_started = True
            self.state = self.STATE_WAIT_ARM_SERVER
            return

        goal_msg = PickBox.Goal()
        goal_msg.side = "RESTORE"

        self.get_logger().info("Sending RESTORE goal to robot arm action server.")
        send_goal_future = self.arm_client.send_goal_async(
            goal_msg,
            feedback_callback=self.restore_feedback_cb
        )
        send_goal_future.add_done_callback(self.restore_goal_response_cb)

        self.restore_goal_sent = True
        self.state = self.STATE_RESTORE_RUNNING

    def restore_feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f"[RESTORE feedback] step={fb.current_step} progress={fb.progress:.2f}"
        )

    def restore_goal_response_cb(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.restore_goal_done = True
            self.restore_goal_success = False
            self.restore_result_message = f"Failed to send restore goal: {e}"
            self.state = self.STATE_RESTORE_FAILED
            self.get_logger().error(self.restore_result_message)
            return

        if not goal_handle.accepted:
            self.restore_goal_done = True
            self.restore_goal_success = False
            self.restore_result_message = "Restore goal was rejected by action server."
            self.state = self.STATE_RESTORE_FAILED
            self.get_logger().error(self.restore_result_message)
            return

        self.get_logger().info("Restore goal accepted.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.restore_result_cb)

    def restore_result_cb(self, future):
        try:
            result_wrap = future.result()
            result = result_wrap.result
        except Exception as e:
            self.restore_goal_done = True
            self.restore_goal_success = False
            self.restore_result_message = f"Failed to get restore result: {e}"
            self.state = self.STATE_RESTORE_FAILED
            self.get_logger().error(self.restore_result_message)
            return

        self.restore_goal_done = True
        self.restore_goal_success = bool(result.success)
        self.restore_result_message = str(result.message)

        if self.restore_goal_success:
            self.state = self.STATE_RESTORE_DONE
            self.get_logger().info(f"Restore completed: {self.restore_result_message}")
        else:
            self.state = self.STATE_RESTORE_FAILED
            self.get_logger().error(f"Restore failed: {self.restore_result_message}")

    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]

        fy0 = int(h * self.follow_roi_y_start)
        fx0 = int(w * self.follow_roi_x_min_ratio)
        fx1 = int(w * self.follow_roi_x_max_ratio)

        fy0 = max(0, min(fy0, h - 1))
        fx0 = max(0, min(fx0, w - 1))
        fx1 = max(fx0 + 1, min(fx1, w))

        follow_roi = frame[fy0:h, fx0:fx1]
        follow_mask = build_white_mask(follow_roi, self.h_low, self.s_low, self.v_low, self.h_high, self.s_high, self.v_high)
        M = cv2.moments(follow_mask)
        area = M["m00"]

        bh = int(h * self.bottom_strip_height_ratio)
        by0 = max(0, h - bh)
        bottom_roi = frame[by0:h, 0:w]

        if self.branch_side == 'LEFT':
            side_x1 = 0
            side_x2 = int(w * self.side_roi_x_ratio)
            side_bottom = bottom_roi[:, 0:side_x2]
        else:
            side_x1 = int(w * (1.0 - self.side_roi_x_ratio))
            side_x2 = w
            side_bottom = bottom_roi[:, side_x1:w]

        side_mask = build_white_mask(side_bottom, self.h_low, self.s_low, self.v_low, self.h_high, self.s_high, self.v_high)
        Ms = cv2.moments(side_mask)
        side_area = Ms["m00"]

        marker_found, marker_center, marker_mode, green_mask, blue_mask, blue_roi_box = self.detect_marker(frame)

        twist = Twist()

        if self.state in [self.STATE_ALIGN_MARKER, self.STATE_APPROACH_MARKER]:
            if self.front_range_valid and self.front_range <= self.front_stop_distance:
                self.state = self.STATE_STOPPED
                self.get_logger().info(
                    f"Target reached by range sensor. Distance = {self.front_range:.3f} m. Robot stopped."
                )

        if self.state == self.STATE_INITIAL_TURN:
            if not self.initial_turn_started:
                self.turn_start_time = self.get_clock().now()
                self.initial_turn_started = True
                self.get_logger().info("Starting initial 90-degree left turn.")

            twist.linear.x = 0.0
            twist.angular.z = self.turn_left_angular_speed

            elapsed = (self.get_clock().now() - self.turn_start_time).nanoseconds / 1e9
            if elapsed >= self.turn_left_90_time:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.state = self.STATE_INITIAL_WAIT
                self.wait_start_time = self.get_clock().now()
                self.get_logger().info("Initial left turn complete.")

        elif self.state == self.STATE_INITIAL_WAIT:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

            elapsed = (self.get_clock().now() - self.wait_start_time).nanoseconds / 1e9
            if elapsed >= self.post_turn_wait_time:
                self.state = self.STATE_FIND_LINE_AFTER_INITIAL_TURN
                self.get_logger().info("Searching for straight white line in center/lower follow ROI.")

        elif self.state == self.STATE_FIND_LINE_AFTER_INITIAL_TURN:
            if area > self.startup_line_detect_min_area:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.state = self.STATE_FOLLOW_LINE_TO_BRANCH
                self.side_branch_counter = 0
                self.branch_count = 0
                self.branch_currently_visible = False
                self.get_logger().info(
                    f"Straight line detected. Starting line follow. follow_area={int(area)}"
                )
            else:
                twist.linear.x = 0.0
                twist.angular.z = self.search_angular

        elif self.state == self.STATE_FOLLOW_LINE_TO_BRANCH:
            self.follow_line(area, M, w // 2, twist, x_offset=fx0)

            main_line_visible = area > self.min_area

            if side_area > self.bottom_branch_min_area:
                self.side_branch_counter += 1
            else:
                self.side_branch_counter = 0

            stable_branch_visible = self.side_branch_counter >= self.junction_confirm_frames

            if self.require_main_line_for_branch:
                stable_branch_visible = stable_branch_visible and main_line_visible

            if stable_branch_visible and not self.branch_currently_visible:
                self.branch_currently_visible = True
                self.branch_count += 1
                self.get_logger().info(
                    f"Detected side branch event #{self.branch_count} on {self.branch_side}"
                )

                if self.branch_count >= self.target_branch_index:
                    self.state = self.STATE_MOVE_FORWARD_BEFORE_TURN
                    self.pre_turn_start_time = self.get_clock().now()
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.get_logger().info(
                        f"Target branch #{self.branch_count} reached. "
                        f"Moving forward {self.pre_turn_distance:.2f} m before turn."
                    )

            elif not stable_branch_visible:
                self.branch_currently_visible = False

        elif self.state == self.STATE_MOVE_FORWARD_BEFORE_TURN:
            twist.linear.x = self.pre_turn_speed
            twist.angular.z = 0.0

            elapsed = (self.get_clock().now() - self.pre_turn_start_time).nanoseconds / 1e9
            if elapsed >= self.pre_turn_time:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.state = self.STATE_TURN_AT_BRANCH
                self.turn_start_time = self.get_clock().now()
                self.get_logger().info("Reached junction center. Turning now.")

        elif self.state == self.STATE_TURN_AT_BRANCH:
            twist.linear.x = 0.0
            twist.angular.z = (
                self.junction_turn_speed if self.branch_side == 'LEFT'
                else -self.junction_turn_speed
            )

            elapsed = (self.get_clock().now() - self.turn_start_time).nanoseconds / 1e9
            if elapsed >= self.junction_turn_90_time:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.state = self.STATE_POST_BRANCH_WAIT
                self.wait_start_time = self.get_clock().now()
                self.get_logger().info("Branch turn complete.")

        elif self.state == self.STATE_POST_BRANCH_WAIT:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

            elapsed = (self.get_clock().now() - self.wait_start_time).nanoseconds / 1e9
            if elapsed >= self.post_turn_wait_time:
                self.state = self.STATE_FOLLOW_LINE_AFTER_BRANCH
                self.marker_seen_counter = 0
                self.get_logger().info("Following line after branch turn.")

        elif self.state == self.STATE_FOLLOW_LINE_AFTER_BRANCH:
            self.follow_line(area, M, w // 2, twist, x_offset=fx0)

            if self.detect_marker_after_branch and marker_found:
                self.marker_seen_counter += 1
            else:
                self.marker_seen_counter = 0

            if self.marker_seen_counter >= self.marker_confirm_frames:
                self.state = self.STATE_ALIGN_MARKER
                self.approach_marker_seen_once = False
                self.approach_marker_lost_counter = 0
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.get_logger().info(f"Marker detected using {marker_mode}. Switching to ALIGN_MARKER.")

        elif self.state == self.STATE_ALIGN_MARKER:
            if marker_found:
                self.align_to_marker(marker_center, w, twist, allow_forward=False)
                error_x = float(marker_center[0] - (w // 2))
                if abs(error_x) <= self.marker_align_tolerance_px:
                    self.state = self.STATE_APPROACH_MARKER
                    self.approach_marker_seen_once = True
                    self.approach_marker_lost_counter = 0
                    self.get_logger().info(f"Marker aligned using {marker_mode}. Switching to APPROACH_MARKER.")
            else:
                twist.linear.x = 0.0
                twist.angular.z = self.marker_search_angular

        elif self.state == self.STATE_APPROACH_MARKER:
            if marker_found:
                self.approach_marker_seen_once = True
                self.approach_marker_lost_counter = 0

                approach_speed = self.marker_approach_speed
                if self.front_range_valid and self.front_range <= self.front_slow_distance:
                    approach_speed = min(approach_speed, 0.045)

                error_x = float(marker_center[0] - (w // 2))
                ang = -self.marker_kp_align * error_x
                ang = self.clamp(ang, -self.marker_max_angular, self.marker_max_angular)

                twist.linear.x = approach_speed
                twist.angular.z = ang
            else:
                # Physical-robot fix:
                # when very close, marker can disappear from view.
                # Instead of rotating forever, stop and finish if we had already started approach.
                if self.approach_marker_seen_once:
                    self.approach_marker_lost_counter += 1
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

                    if self.approach_marker_lost_counter >= self.approach_marker_lost_limit:
                        self.state = self.STATE_STOPPED
                        self.get_logger().info(
                            "Marker lost after close approach. Assuming target reached. Robot stopped."
                        )
                else:
                    twist.linear.x = 0.0
                    twist.angular.z = self.marker_search_angular

        elif self.state == self.STATE_STOPPED:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            if self.trigger_restore_after_stop:
                self.try_start_restore_action()

        elif self.state == self.STATE_WAIT_ARM_SERVER:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            if self.arm_client.server_is_ready():
                self.get_logger().info("Arm action server is now ready.")
                self.try_start_restore_action()

        elif self.state == self.STATE_RESTORE_RUNNING:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        elif self.state == self.STATE_RESTORE_DONE:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.task3_done = True
            self.publish_task_done()

        elif self.state == self.STATE_RESTORE_FAILED:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

        vis = frame.copy()

        cv2.rectangle(vis, (fx0, fy0), (fx1 - 1, h - 1), (0, 255, 0), 2)
        cv2.putText(
            vis, "FOLLOW ROI", (fx0 + 5, max(25, fy0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        cv2.rectangle(vis, (side_x1, by0), (side_x2 - 1, h - 1), (255, 255, 0), 2)
        cv2.putText(
            vis, f"{self.branch_side} BOTTOM BRANCH ROI", (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
        )

        bx0, byy0, bx1, byy1 = blue_roi_box
        cv2.rectangle(vis, (bx0, byy0), (bx1 - 1, byy1 - 1), (255, 0, 0), 2)
        cv2.putText(
            vis, "BLUE ROI", (bx0 + 5, max(20, byy0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
        )

        if area > self.min_area:
            cx_vis = fx0 + int(M["m10"] / area)
            cy_vis = fy0 + (h - fy0) // 2
            cv2.circle(vis, (cx_vis, cy_vis), 8, (0, 255, 255), -1)

        cv2.line(vis, (w // 2, 0), (w // 2, h - 1), (0, 0, 255), 1)
        cv2.line(vis, (0, h // 2), (w - 1, h // 2), (0, 0, 255), 1)

        for p in self.last_green_centers:
            cv2.circle(vis, p, 14, (0, 255, 0), 2)
            cv2.circle(vis, p, 3, (0, 255, 0), -1)

        for p in self.last_blue_centers:
            cv2.circle(vis, p, 14, (255, 0, 0), 2)
            cv2.circle(vis, p, 3, (255, 0, 0), -1)

        if len(self.last_green_centers) == 2:
            cv2.line(vis, self.last_green_centers[0], self.last_green_centers[1], (0, 255, 0), 2)

        if len(self.last_blue_centers) == 2:
            cv2.line(vis, self.last_blue_centers[0], self.last_blue_centers[1], (255, 0, 0), 2)

        if self.last_marker_center is not None:
            cv2.circle(vis, self.last_marker_center, 10, (0, 255, 255), -1)
            cv2.putText(
                vis, f"MARKER CENTER ({self.last_marker_mode})",
                (self.last_marker_center[0] + 10, self.last_marker_center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
            )

        cv2.putText(
            vis, f"STATE: {self.state}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

        cv2.putText(
            vis, f"follow_area={int(area)} side_area={int(side_area)}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        cv2.putText(
            vis, f"branch_count={self.branch_count} target={self.target_branch_index}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 255, 180), 2
        )

        cv2.putText(
            vis, f"marker_found={marker_found} mode={marker_mode} counter={self.marker_seen_counter}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 220, 150), 2
        )

        front_txt = "front=NA"
        if self.front_range_valid:
            front_txt = f"front={self.front_range:.3f} m"

        cv2.putText(
            vis, front_txt, (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2
        )

        cv2.putText(
            vis, f"cmd v={twist.linear.x:.2f} w={twist.angular.z:.2f}", (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        cv2.putText(
            vis,
            f"approach_seen={self.approach_marker_seen_once} lost={self.approach_marker_lost_counter}",
            (10, 210),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2
        )

        if self.restore_goal_sent:
            cv2.putText(
                vis, f"RESTORE sent | done={self.restore_goal_done} ok={self.restore_goal_success}",
                (10, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2
            )

        if self.task3_done_published:
            cv2.putText(
                vis, "TASK3 STATUS: DONE",
                (10, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

        # cv2.imshow("task3_camera", vis)

        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     self.get_logger().info("Quit requested.")
        #     self.stop_robot()
        #     rclpy.shutdown()
        #     cv2.destroyAllWindows()
        #     return

        self.frame_count += 1
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds > 1_000_000_000:
            self.last_log_time = now
            self.get_logger().info(
                f"fps~{self.frame_count} state={self.state} "
                f"follow_area={int(area)} side_area={int(side_area)} "
                f"branch_count={self.branch_count}/{self.target_branch_index} "
                f"marker_found={marker_found} mode={marker_mode} "
                f"front={'NA' if not self.front_range_valid else f'{self.front_range:.3f}'} "
                f"approach_seen={self.approach_marker_seen_once} "
                f"approach_lost={self.approach_marker_lost_counter} "
                f"restore_sent={self.restore_goal_sent} "
                f"task3_done_pub={self.task3_done_published} "
                f"cmd(v,w)=({twist.linear.x:.2f},{twist.angular.z:.2f})"
            )
            self.frame_count = 0


def main():
    rclpy.init()
    node = Task3Node()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass

        node.destroy_node()
        # cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()