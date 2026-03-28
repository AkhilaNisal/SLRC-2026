#!/usr/bin/env python3

# starting pos in arena :translation 0.75 -1.05 0

import math
from collections import deque

import gpiod
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.action import ActionClient
from rclpy.duration import Duration

from sensor_msgs.msg import Image, Range
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge

import cv2
import numpy as np

from robocop_pkg.line_detection_utils import build_white_mask

from robot_arm_interfaces.action import PickBox


class WhiteLineFollowerWithBoxVisit(Node):
    def __init__(self):
        super().__init__('task2_with_arm')

        # =========================
        # Topics
        # =========================
        self.declare_parameter('image_topic', '/camera/image/image_color')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('left_range_topic', '/robocop/ds_left')
        self.declare_parameter('right_range_topic', '/robocop/ds_right')
        self.declare_parameter('front_range_topic', '/robocop/ds_front')
        self.declare_parameter('task2_status_topic', '/task2/status')
        self.declare_parameter('gyro_angle_topic', '/gyro_angle')

        # =========================
        # Motion control
        # =========================
        self.declare_parameter('forward_speed', 0.25)
        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('kp', 0.004)
        self.declare_parameter('max_angular', 1.2)

        self.declare_parameter('extra_forward_distance', 0.19)
        self.declare_parameter('post_turn_wait_time', 1.0)

        self.declare_parameter('search_linear', 0.04)
        self.declare_parameter('search_angular', 0.35)

        # =========================
        # Gyro turning control
        # =========================
        self.declare_parameter('turn_angle_90_deg', 90.0)
        self.declare_parameter('turn_tolerance_deg', 3.0)
        self.declare_parameter('turn_kp', 0.018)
        self.declare_parameter('turn_min_speed', 0.22)
        self.declare_parameter('turn_max_speed', 0.9)
        self.declare_parameter('turn_timeout_sec', 6.0)

        # =========================
        # White detection
        # =========================
        self.declare_parameter('roi_y_start', 0.70)
        self.declare_parameter('min_area', 5000)

        self.declare_parameter('bottom_strip_height_ratio', 0.14)
        self.declare_parameter('bottom_min_area', 2500)
        self.declare_parameter('line_gone_frames', 5)

        self.declare_parameter('h_low', 0)
        self.declare_parameter('s_low', 0)
        self.declare_parameter('v_low', 180)
        self.declare_parameter('h_high', 180)
        self.declare_parameter('s_high', 70)
        self.declare_parameter('v_high', 255)

        # =========================
        # Red box detection
        # =========================
        self.declare_parameter('red_h1_low', 0)
        self.declare_parameter('red_h1_high', 12)
        self.declare_parameter('red_h2_low', 165)
        self.declare_parameter('red_h2_high', 180)
        self.declare_parameter('red_s_low', 70)
        self.declare_parameter('red_v_low', 40)
        self.declare_parameter('red_min_area', 200)

        # red centering while approaching
        self.declare_parameter('red_kp', 0.0045)
        self.declare_parameter('red_max_angular', 1.0)
        self.declare_parameter('red_search_angular', 0.35)
        self.declare_parameter('red_lost_frames_limit', 12)

        # =========================
        # Distance sensing / filter
        # Normal-mode VL53L0X runs at ~20 Hz (no measurement_timing_budget set).
        # Alpha values here are tuned for that rate — lower than accuracy-mode values
        # to achieve equivalent per-second smoothing with more frames available.
        # =========================
        self.declare_parameter('range_filter_alpha', 0.15)          # EMA for right/front (20 Hz)
        self.declare_parameter('left_range_filter_alpha', 0.25)     # slightly faster for left
        self.declare_parameter('print_distances_every_frame', False)

        # =========================
        # Delta-based ToF box detection while following line
        #
        # Instead of comparing against a fixed absolute distance, a rolling-median
        # baseline is maintained per side.  When the filtered reading drops by more
        # than tof_delta_threshold below that baseline the box confirm counter
        # increments; reaching tof_confirm_frames consecutive hits triggers detection.
        #
        # This is robust to different arena layouts (no need to hard-code wall
        # distance) and rejects single-frame spikes because the baseline is a median.
        # =========================
        self.declare_parameter('tof_delta_threshold', 0.05)   # 5 cm drop from baseline → box
        self.declare_parameter('tof_baseline_window', 20)     # frames kept for rolling median
        self.declare_parameter('tof_confirm_frames', 5)       # consecutive delta-hits to confirm

        # ignore box search at first line-follow start
        self.declare_parameter('startup_box_ignore_distance', 0.25)

        # ignore same box after handling
        self.declare_parameter('same_box_ignore_distance', 0.35)

        # front obstacle / wall stop — ~0.25 s at 20 Hz
        self.declare_parameter('front_obstacle_stop_distance', 0.20)
        self.declare_parameter('front_obstacle_stop_frames', 5)

        # =========================
        # Red LED blink on box detection
        # Uses gpiod (same library as stepper motor driver).
        # blink_half_period_sec is the ON or OFF duration of each half-cycle.
        # Two full blinks = 4 transitions (ON→OFF→ON→OFF) over 4×period seconds.
        # =========================
        self.declare_parameter('led_chip_name', 'gpiochip4')
        self.declare_parameter('led_pin', 26)
        self.declare_parameter('blink_half_period_sec', 0.3)

        # =========================
        # Box visit maneuver
        # =========================
        self.declare_parameter('box_forward_before_turn_distance', 0.07)  # 5 cm forward along line before turning
        self.declare_parameter('box_approach_speed', 0.08)
        self.declare_parameter('box_stop_distance', 0.15)        # compatibility only
        self.declare_parameter('box_front_stop_distance', 0.25)  # actual stop distance
        self.declare_parameter('box_stop_frames', 1)
        self.declare_parameter('box_return_speed', 0.12)
        self.declare_parameter('box_drive_timeout_sec', 8.0)

        # reverse after pick
        self.declare_parameter('reverse_after_pick_speed', 0.15)
        self.declare_parameter('reverse_after_pick_distance', 0.60)

        # =========================
        # Action client / pickup behavior
        # =========================
        self.declare_parameter('pick_action_name', '/pick_box')
        self.declare_parameter('pick_retry_limit', 5)
        self.declare_parameter('pick_goal_send_once', True)

        # =========================
        # Task 2 finish behavior
        # =========================
        self.declare_parameter('target_box_count', 6)
        self.declare_parameter('task2_finish_wall_distance', 0.4)
        self.declare_parameter('task2_finish_wall_frames', 1)
        self.declare_parameter('task2_finish_forward_speed', 0.08)

        # Debug visualization (disable on headless robot)
        self.declare_parameter('debug', False)

        # =========================
        # Read params
        # =========================
        self.image_topic = self.get_parameter('image_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.left_range_topic = self.get_parameter('left_range_topic').value
        self.right_range_topic = self.get_parameter('right_range_topic').value
        self.front_range_topic = self.get_parameter('front_range_topic').value
        self.task2_status_topic = self.get_parameter('task2_status_topic').value
        self.gyro_angle_topic = self.get_parameter('gyro_angle_topic').value

        self.forward_speed = float(self.get_parameter('forward_speed').value)
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.kp = float(self.get_parameter('kp').value)
        self.max_angular = float(self.get_parameter('max_angular').value)

        self.extra_forward_distance = float(self.get_parameter('extra_forward_distance').value)
        self.extra_forward_time = (
            self.extra_forward_distance / self.forward_speed
            if self.forward_speed > 1e-6 else 0.0
        )

        self.post_turn_wait_time = float(self.get_parameter('post_turn_wait_time').value)

        self.search_linear = float(self.get_parameter('search_linear').value)
        self.search_angular = float(self.get_parameter('search_angular').value)

        self.turn_angle_90_deg = float(self.get_parameter('turn_angle_90_deg').value)
        self.turn_tolerance_deg = float(self.get_parameter('turn_tolerance_deg').value)
        self.turn_kp = float(self.get_parameter('turn_kp').value)
        self.turn_min_speed = float(self.get_parameter('turn_min_speed').value)
        self.turn_max_speed = float(self.get_parameter('turn_max_speed').value)
        self.turn_timeout_sec = float(self.get_parameter('turn_timeout_sec').value)

        self.roi_y_start = float(self.get_parameter('roi_y_start').value)
        self.min_area = int(self.get_parameter('min_area').value)

        self.bottom_strip_height_ratio = float(self.get_parameter('bottom_strip_height_ratio').value)
        self.bottom_min_area = int(self.get_parameter('bottom_min_area').value)
        self.line_gone_frames = int(self.get_parameter('line_gone_frames').value)

        self.h_low = int(self.get_parameter('h_low').value)
        self.s_low = int(self.get_parameter('s_low').value)
        self.v_low = int(self.get_parameter('v_low').value)
        self.h_high = int(self.get_parameter('h_high').value)
        self.s_high = int(self.get_parameter('s_high').value)
        self.v_high = int(self.get_parameter('v_high').value)

        self.red_h1_low = int(self.get_parameter('red_h1_low').value)
        self.red_h1_high = int(self.get_parameter('red_h1_high').value)
        self.red_h2_low = int(self.get_parameter('red_h2_low').value)
        self.red_h2_high = int(self.get_parameter('red_h2_high').value)
        self.red_s_low = int(self.get_parameter('red_s_low').value)
        self.red_v_low = int(self.get_parameter('red_v_low').value)
        self.red_min_area = int(self.get_parameter('red_min_area').value)
        self.red_kp = float(self.get_parameter('red_kp').value)
        self.red_max_angular = float(self.get_parameter('red_max_angular').value)
        self.red_search_angular = float(self.get_parameter('red_search_angular').value)
        self.red_lost_frames_limit = int(self.get_parameter('red_lost_frames_limit').value)

        self.range_filter_alpha = float(self.get_parameter('range_filter_alpha').value)
        self.left_range_filter_alpha = float(self.get_parameter('left_range_filter_alpha').value)
        self.print_distances_every_frame = bool(self.get_parameter('print_distances_every_frame').value)

        self.tof_delta_threshold = float(self.get_parameter('tof_delta_threshold').value)
        self.tof_baseline_window = int(self.get_parameter('tof_baseline_window').value)
        self.tof_confirm_frames = int(self.get_parameter('tof_confirm_frames').value)

        self.led_chip_name = str(self.get_parameter('led_chip_name').value)
        self.led_pin = int(self.get_parameter('led_pin').value)
        self.blink_half_period_sec = float(self.get_parameter('blink_half_period_sec').value)

        self.startup_box_ignore_distance = float(self.get_parameter('startup_box_ignore_distance').value)
        self.startup_box_ignore_time = (
            self.startup_box_ignore_distance / self.linear_speed
            if self.linear_speed > 1e-6 else 0.0
        )

        self.same_box_ignore_distance = float(self.get_parameter('same_box_ignore_distance').value)
        self.same_box_ignore_time = (
            self.same_box_ignore_distance / self.linear_speed
            if self.linear_speed > 1e-6 else 0.0
        )

        self.front_obstacle_stop_distance = float(self.get_parameter('front_obstacle_stop_distance').value)
        self.front_obstacle_stop_frames = int(self.get_parameter('front_obstacle_stop_frames').value)

        self.box_forward_before_turn_distance = float(self.get_parameter('box_forward_before_turn_distance').value)
        self.box_approach_speed = float(self.get_parameter('box_approach_speed').value)
        self.box_stop_distance = float(self.get_parameter('box_stop_distance').value)
        self.box_front_stop_distance = float(self.get_parameter('box_front_stop_distance').value)
        self.box_stop_frames = int(self.get_parameter('box_stop_frames').value)
        self.box_return_speed = float(self.get_parameter('box_return_speed').value)
        self.box_drive_timeout_sec = float(self.get_parameter('box_drive_timeout_sec').value)

        self.reverse_after_pick_speed = float(self.get_parameter('reverse_after_pick_speed').value)
        self.reverse_after_pick_distance = float(self.get_parameter('reverse_after_pick_distance').value)
        self.reverse_after_pick_time = (
            self.reverse_after_pick_distance / self.reverse_after_pick_speed
            if self.reverse_after_pick_speed > 1e-6 else 0.0
        )

        self.pick_action_name = str(self.get_parameter('pick_action_name').value)
        self.pick_retry_limit = int(self.get_parameter('pick_retry_limit').value)
        self.pick_goal_send_once = bool(self.get_parameter('pick_goal_send_once').value)

        self.target_box_count = int(self.get_parameter('target_box_count').value)
        self.task2_finish_wall_distance = float(self.get_parameter('task2_finish_wall_distance').value)
        self.task2_finish_wall_frames = int(self.get_parameter('task2_finish_wall_frames').value)
        self.task2_finish_forward_speed = float(self.get_parameter('task2_finish_forward_speed').value)
        self.debug = bool(self.get_parameter('debug').value)

        # ROS interfaces
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_cb, qos_profile_sensor_data
        )
        self.left_range_sub = self.create_subscription(
            Range, self.left_range_topic, self.left_range_cb, qos_profile_sensor_data
        )
        self.right_range_sub = self.create_subscription(
            Range, self.right_range_topic, self.right_range_cb, qos_profile_sensor_data
        )
        self.front_range_sub = self.create_subscription(
            Range, self.front_range_topic, self.front_range_cb, qos_profile_sensor_data
        )
        self.gyro_angle_sub = self.create_subscription(
            Float32, self.gyro_angle_topic, self.gyro_angle_cb, qos_profile_sensor_data
        )

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.task_status_pub = self.create_publisher(String, self.task2_status_topic, 10)

        self.pick_client = ActionClient(self, PickBox, self.pick_action_name)

        self.pick_server_ready_logged = False
        self.server_check_timer = self.create_timer(0.5, self.check_pick_server)

        # States
        self.STATE_LINE_CROSS_APPROACH = 'LINE_CROSS_APPROACH'
        self.STATE_LINE_CROSS_WAIT_DISAPPEAR = 'LINE_CROSS_WAIT_DISAPPEAR'
        self.STATE_LINE_CROSS_EXTRA_FORWARD = 'LINE_CROSS_EXTRA_FORWARD'
        self.STATE_LINE_CROSS_TURN = 'LINE_CROSS_TURN'
        self.STATE_LINE_CROSS_POST_WAIT = 'LINE_CROSS_POST_WAIT'

        self.STATE_FOLLOW_LINE = 'FOLLOW_LINE'

        self.STATE_BOX_FORWARD_BEFORE_TURN = 'BOX_FORWARD_BEFORE_TURN'
        self.STATE_BOX_TURN_TO_BOX = 'BOX_TURN_TO_BOX'
        self.STATE_BOX_DRIVE_TO_BOX = 'BOX_DRIVE_TO_BOX'
        self.STATE_BOX_REQUEST_PICK = 'BOX_REQUEST_PICK'
        self.STATE_BOX_WAIT_PICK_RESULT = 'BOX_WAIT_PICK_RESULT'
        self.STATE_BOX_PICK_FAILED = 'BOX_PICK_FAILED'
        self.STATE_BOX_REVERSE_AFTER_PICK = 'BOX_REVERSE_AFTER_PICK'
        self.STATE_BOX_TURN_TO_RESUME = 'BOX_TURN_TO_RESUME'
        # Temporary state: robot is stopped while LED blinks twice to signal detection.
        self.STATE_BOX_BLINK = 'BOX_BLINK'
        self.STATE_TASK2_DONE = 'TASK2_DONE'

        self.state = self.STATE_LINE_CROSS_APPROACH

        # line-cross reusable context
        self.line_cross_turn_direction = +1.0
        self.line_cross_next_state = self.STATE_FOLLOW_LINE
        self.line_seen = False
        self.line_gone_counter = 0
        self.extra_forward_start_time = None
        self.post_turn_wait_start_time = None
        self.line_cross_speed = self.forward_speed

        # gyro turn state
        self.current_yaw_deg = None
        self.gyro_ready = False
        self.turn_active = False
        self.turn_target_deg = None
        self.turn_next_state = None
        self.turn_start_time = None

        # box visit state
        self.active_box_side = None
        self.box_stop_counter = 0
        self.left_box_count = 0
        self.right_box_count = 0
        self.boxes_completed = 0
        self.box_drive_start_time = None
        self.red_lost_counter = 0
        self.reverse_start_time = None
        self.box_forward_before_turn_start_time = None

        # ignore same-side box detection after one pickup
        self.ignore_box_side = None
        self.ignore_box_until_time = None

        # ignore all box detection only at the first time line following starts
        self.startup_box_ignore_active = False
        self.startup_box_ignore_until_time = None
        self.startup_box_ignore_used = False

        # front obstacle termination
        self.front_obstacle_counter = 0

        # action state
        self.pick_goal_sent = False
        self.pick_in_progress = False
        self.pick_result_ready = False
        self.pick_result_success = False
        self.pick_result_message = ""
        self.pick_feedback_text = ""
        self.pick_retry_count = 0
        self.current_goal_handle = None
        self.pick_failed_latched = False

        # sensing
        self.left_range_raw = math.inf
        self.right_range_raw = math.inf
        self.front_range_raw = math.inf

        self.left_range = math.inf
        self.right_range = math.inf
        self.front_range = math.inf

        self.measurement_started = False

        # Delta-based box detection counters (consecutive confirm frames per side)
        self.left_detect_counter = 0
        self.right_detect_counter = 0

        # Rolling-median baseline buffers — one per side (see _compute_delta)
        self.left_baseline_buf: deque = deque(maxlen=self.tof_baseline_window)
        self.right_baseline_buf: deque = deque(maxlen=self.tof_baseline_window)

        # Debug sparkline history (separate from baseline — always updated, finite-invalid clamped to max)
        self.left_vis_buf: deque = deque(maxlen=120)
        self.right_vis_buf: deque = deque(maxlen=120)

        # LED blink state
        self._led_line = None          # gpiod line handle, set by _init_led
        self._blink_count = 0          # how many half-periods have elapsed
        self._blink_timer = None       # ROS timer object, created per blink sequence
        self._blink_detected_side = None

        self._init_led()

        self.finish_wall_counter = 0
        self.task2_done = False
        self.task2_done_published = False

        # debug
        self.frame_count = 0
        self.last_log_time = self.get_clock().now()

        if self.debug:
            cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
            self.get_logger().info("Debug windows enabled. Press 'q' to quit.")
        else:
            self.get_logger().info("Debug windows disabled (headless mode). Set debug:=true to enable.")

        self.configure_line_cross_sequence(
            speed=self.forward_speed,
            turn_direction=+1.0,
            next_state=self.STATE_FOLLOW_LINE
        )

        self.get_logger().info(
            "Started task2_with_arm: delta-ToF box detection (5 cm drop from rolling median), "
            "red LED blink×2 on detection, normal-mode ToF filter settings."
        )

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def fmt_range(x: float) -> str:
        if math.isinf(x):
            return "inf"
        if math.isnan(x):
            return "nan"
        return f"{x:.3f}"

    @staticmethod
    def normalize_angle_deg(angle: float) -> float:
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    @classmethod
    def angle_diff_deg(cls, target: float, current: float) -> float:
        return cls.normalize_angle_deg(target - current)

    def valid_range(self, x: float) -> bool:
        return not math.isinf(x) and not math.isnan(x) and x > 0.0

    def display_range(self, x: float) -> str:
        if not self.measurement_started:
            return "OFF"
        return self.fmt_range(x)

    def side_sign(self, side: str) -> float:
        return 1.0 if side == 'LEFT' else -1.0

    def opposite_side_sign(self, side: str) -> float:
        return -self.side_sign(side)

    def current_side_range(self) -> float:
        if self.active_box_side == 'LEFT':
            return self.left_range
        return self.right_range

    def low_pass_filter(self, previous: float, current: float, alpha: float) -> float:
        if math.isinf(previous) or math.isnan(previous):
            return current
        return alpha * current + (1.0 - alpha) * previous

    def left_range_cb(self, msg: Range):
        raw = float(msg.range)
        self.left_range_raw = raw
        self.left_range = self.low_pass_filter(self.left_range, raw, self.left_range_filter_alpha)
        self.left_vis_buf.append(self.left_range if self.valid_range(self.left_range) else 2.0)

    def right_range_cb(self, msg: Range):
        raw = float(msg.range)
        self.right_range_raw = raw
        self.right_range = self.low_pass_filter(self.right_range, raw, self.range_filter_alpha)
        self.right_vis_buf.append(self.right_range if self.valid_range(self.right_range) else 2.0)

    def front_range_cb(self, msg: Range):
        raw = float(msg.range)
        self.front_range_raw = raw
        self.front_range = self.low_pass_filter(self.front_range, raw, self.range_filter_alpha)

    def gyro_angle_cb(self, msg: Float32):
        self.current_yaw_deg = self.normalize_angle_deg(float(msg.data))
        self.gyro_ready = True

    def build_red_mask(self, bgr_img):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

        lower1 = np.array([self.red_h1_low, self.red_s_low, self.red_v_low], dtype=np.uint8)
        upper1 = np.array([self.red_h1_high, 255, 255], dtype=np.uint8)

        lower2 = np.array([self.red_h2_low, self.red_s_low, self.red_v_low], dtype=np.uint8)
        upper2 = np.array([self.red_h2_high, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def detect_red_box(self, frame):
        mask = self.build_red_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0.0

        for c in contours:
            a = cv2.contourArea(c)
            if a < self.red_min_area:
                continue
            if a > best_area:
                best_area = a
                best = c

        if best is None:
            return False, None, None, 0.0, None, mask

        x, y, bw, bh = cv2.boundingRect(best)
        M = cv2.moments(best)
        if M["m00"] <= 0:
            return False, None, None, 0.0, None, mask

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return True, cx, cy, float(best_area), (x, y, bw, bh), mask

    def start_measurement(self):
        if self.measurement_started:
            return
        self.measurement_started = True
        self.left_detect_counter = 0
        self.right_detect_counter = 0
        self.finish_wall_counter = 0
        self.front_obstacle_counter = 0
        self.get_logger().info("Started box detection during white-line following.")

    def start_startup_box_ignore(self):
        self.startup_box_ignore_active = True
        self.startup_box_ignore_used = True
        self.startup_box_ignore_until_time = self.get_clock().now() + Duration(
            seconds=self.startup_box_ignore_time
        )
        self.get_logger().info(
            f"Ignoring all box detection for first {self.startup_box_ignore_distance:.2f} m "
            f"({self.startup_box_ignore_time:.2f} s) after initial white-line start."
        )

    def startup_ignore_active_now(self) -> bool:
        if not self.startup_box_ignore_active:
            return False
        if self.startup_box_ignore_until_time is None:
            return False

        now = self.get_clock().now()
        if now < self.startup_box_ignore_until_time:
            return True

        self.startup_box_ignore_active = False
        self.startup_box_ignore_until_time = None
        self.get_logger().info("Initial box-search ignore finished. Normal side TOF box detection enabled.")
        return False

    def reset_box_detection_counters(self):
        self.left_detect_counter = 0
        self.right_detect_counter = 0

    def same_side_is_ignored(self, side: str) -> bool:
        if self.ignore_box_side != side:
            return False
        if self.ignore_box_until_time is None:
            return False

        now = self.get_clock().now()
        if now < self.ignore_box_until_time:
            return True

        self.ignore_box_side = None
        self.ignore_box_until_time = None
        return False

    def start_same_box_ignore(self, side: str):
        self.ignore_box_side = side
        self.ignore_box_until_time = self.get_clock().now() + Duration(
            seconds=self.same_box_ignore_time
        )
        self.get_logger().info(
            f"Ignoring {side} side box detections for about {self.same_box_ignore_distance:.2f} m "
            f"({self.same_box_ignore_time:.2f} s)."
        )

    def front_obstacle_should_stop(self) -> bool:
        hit = self.valid_range(self.front_range) and self.front_range <= self.front_obstacle_stop_distance
        if hit:
            self.front_obstacle_counter += 1
        else:
            self.front_obstacle_counter = 0
        return self.front_obstacle_counter >= self.front_obstacle_stop_frames

    # =========================
    # DELTA-BASED BOX DETECTION
    #
    # Each ToF reading is compared against a rolling median of recent background
    # readings (left_baseline_buf / right_baseline_buf).  A drop of at least
    # tof_delta_threshold (default 5 cm) from the median triggers one confirm
    # frame; tof_confirm_frames consecutive frames are required for a detection.
    #
    # Why median baseline instead of absolute threshold:
    #   - Works at any arena size / robot-to-wall distance without re-tuning.
    #   - Single-frame noise spikes don't move the median.
    #   - Baseline self-updates when no box is present, so drift is handled.
    #
    # Baseline update rule: a new reading is only added to the buffer when it
    # is NOT a detection frame.  This stops the box's reflection from gradually
    # pulling the baseline down and masking itself.
    # =========================
    def _compute_delta(self, reading: float, buf: deque) -> bool:
        """
        Returns True (box present) when `reading` is >= tof_delta_threshold
        below the rolling-median baseline.  Updates the baseline only on
        non-detection frames so the box cannot corrupt its own baseline.
        """
        if not self.valid_range(reading):
            return False

        if len(buf) < 3:
            # Not enough history yet — prime the buffer and withhold judgement.
            buf.append(reading)
            return False

        baseline = float(np.median(list(buf)))
        drop = baseline - reading  # positive when robot passes a closer object

        if drop >= self.tof_delta_threshold:
            # Box detected — do NOT add this reading to baseline
            return True

        # Background reading — update baseline
        buf.append(reading)
        return False

    def choose_box_side(self):
        """
        Called every image frame while in STATE_FOLLOW_LINE.
        Returns 'LEFT', 'RIGHT', or None.
        """
        if not self.measurement_started or self.state != self.STATE_FOLLOW_LINE:
            self.reset_box_detection_counters()
            return None

        left_ignored = self.same_side_is_ignored('LEFT')
        right_ignored = self.same_side_is_ignored('RIGHT')

        if self.startup_ignore_active_now():
            # Still prime baselines during startup ignore, but don't trigger.
            self._compute_delta(self.left_range, self.left_baseline_buf)
            self._compute_delta(self.right_range, self.right_baseline_buf)
            self.reset_box_detection_counters()
            return None

        # --- Left side ---
        if left_ignored:
            # Freely update baseline while ignored so it is fresh when ignore ends.
            if self.valid_range(self.left_range):
                self.left_baseline_buf.append(self.left_range)
            self.left_detect_counter = 0
            left_delta = False
        else:
            left_delta = self._compute_delta(self.left_range, self.left_baseline_buf)

        # --- Right side ---
        if right_ignored:
            if self.valid_range(self.right_range):
                self.right_baseline_buf.append(self.right_range)
            self.right_detect_counter = 0
            right_delta = False
        else:
            right_delta = self._compute_delta(self.right_range, self.right_baseline_buf)

        # --- Update confirm counters (fast reset on miss, no slow-decay needed at 20 Hz) ---
        if left_delta:
            self.left_detect_counter += 1
        else:
            self.left_detect_counter = 0

        if right_delta:
            self.right_detect_counter += 1
        else:
            self.right_detect_counter = 0

        left_ready = self.left_detect_counter >= self.tof_confirm_frames
        right_ready = self.right_detect_counter >= self.tof_confirm_frames

        if self.left_detect_counter > 0 or self.right_detect_counter > 0:
            left_base = (
                float(np.median(list(self.left_baseline_buf)))
                if len(self.left_baseline_buf) >= 3 else float('nan')
            )
            right_base = (
                float(np.median(list(self.right_baseline_buf)))
                if len(self.right_baseline_buf) >= 3 else float('nan')
            )
            self.get_logger().debug(
                f"BoxDetect: L_cnt={self.left_detect_counter}/{self.tof_confirm_frames} "
                f"R_cnt={self.right_detect_counter}/{self.tof_confirm_frames} "
                f"L_filt={self.fmt_range(self.left_range)} L_base={left_base:.3f} "
                f"R_filt={self.fmt_range(self.right_range)} R_base={right_base:.3f}"
            )

        if not left_ready and not right_ready:
            return None

        # Both ready → pick the side with the stronger (more frames) count
        if left_ready and right_ready:
            side = 'LEFT' if self.left_detect_counter >= self.right_detect_counter else 'RIGHT'
        elif left_ready:
            side = 'LEFT'
        else:
            side = 'RIGHT'

        self.get_logger().info(
            f"Box detected on {side}! L_cnt={self.left_detect_counter} "
            f"R_cnt={self.right_detect_counter} "
            f"L_range={self.fmt_range(self.left_range)} "
            f"R_range={self.fmt_range(self.right_range)}"
        )
        self.reset_box_detection_counters()
        return side

    # =========================
    # LED CONTROL
    # =========================
    def _init_led(self):
        """Open the gpiod LED line.  Logs a warning and continues if it fails."""
        try:
            chip = gpiod.Chip(self.led_chip_name)
            self._led_line = chip.get_line(self.led_pin)
            self._led_line.request(
                consumer='task2_led',
                type=gpiod.LINE_REQ_DIR_OUT,
                default_vals=[0]
            )
            self.get_logger().info(
                f"Red LED initialized on {self.led_chip_name} pin {self.led_pin}"
            )
        except Exception as exc:
            self._led_line = None
            self.get_logger().warn(
                f"LED init failed ({self.led_chip_name} pin {self.led_pin}): {exc}. "
                "Box detection will log only — no physical blink."
            )

    def _set_led(self, on: bool):
        """Set the LED on (True) or off (False)."""
        if self._led_line is None:
            return
        try:
            self._led_line.set_value(1 if on else 0)
        except Exception as exc:
            self.get_logger().warn(f"LED set_value error: {exc}")

    def _start_blink(self, detected_side: str):
        """
        Stop the robot, record which side triggered, and start the 2×blink sequence.

        The blink pattern is:  ON → OFF → ON → OFF
        with blink_half_period_sec between each transition (default 0.3 s).
        A ROS timer drives each half-period via _blink_step().
        After the sequence the robot returns to STATE_FOLLOW_LINE.
        """
        # Cancel any leftover blink timer from a previous detection
        if self._blink_timer is not None:
            self._blink_timer.cancel()
            self._blink_timer = None

        self._blink_detected_side = detected_side
        self._blink_count = 0
        self.state = self.STATE_BOX_BLINK

        self._set_led(True)  # first ON immediately
        self._blink_timer = self.create_timer(
            self.blink_half_period_sec, self._blink_step
        )
        self.get_logger().info(
            f"Box detected on {detected_side} side — stopped, blinking red LED ×2."
        )

    def _blink_step(self):
        """
        Timer callback called every blink_half_period_sec.
        Drives the ON→OFF→ON→OFF LED sequence and transitions back to line-follow.
        """
        self._blink_count += 1
        # Transition table after the initial ON:
        #   count 1 → OFF,  count 2 → ON,  count 3 → OFF (done)
        led_on = (self._blink_count % 2 == 0)
        self._set_led(led_on)

        if self._blink_count >= 3:
            # Sequence complete — ensure LED is off
            self._set_led(False)
            self._blink_timer.cancel()
            self._blink_timer = None

            # Ignore this side briefly so we don't immediately re-detect
            if self._blink_detected_side is not None:
                self.start_same_box_ignore(self._blink_detected_side)

            self.reset_box_detection_counters()
            self.state = self.STATE_FOLLOW_LINE
            self.get_logger().info("Blink complete — resuming white-line following.")
    
    def configure_line_cross_sequence(self, speed: float, turn_direction: float, next_state: str):
        self.line_cross_speed = speed
        self.line_cross_turn_direction = turn_direction
        self.line_cross_next_state = next_state
        self.line_seen = False
        self.line_gone_counter = 0
        self.extra_forward_start_time = None
        self.post_turn_wait_start_time = None
        self.turn_active = False
        self.turn_target_deg = None
        self.turn_next_state = None
        self.turn_start_time = None
        self.state = self.STATE_LINE_CROSS_APPROACH

    def start_box_detour(self, side: str):
        self.active_box_side = side
        self.box_stop_counter = 0
        self.box_drive_start_time = None
        self.red_lost_counter = 0
        self.reverse_start_time = None
        self.box_forward_before_turn_start_time = None

        self.pick_goal_sent = False
        self.pick_in_progress = False
        self.pick_result_ready = False
        self.pick_result_success = False
        self.pick_result_message = ""
        self.pick_feedback_text = ""
        self.pick_retry_count = 0
        self.current_goal_handle = None
        self.pick_failed_latched = False

        if side == 'LEFT':
            self.left_box_count += 1
            count = self.left_box_count
        else:
            self.right_box_count += 1
            count = self.right_box_count

        # Drive forward along the white line for box_forward_before_turn_distance
        # before turning 90° toward the box
        if self.box_forward_before_turn_distance > 1e-6:
            self.state = self.STATE_BOX_FORWARD_BEFORE_TURN
            self.box_forward_before_turn_start_time = self.get_clock().now()
            fwd_time = self.box_forward_before_turn_distance / self.linear_speed if self.linear_speed > 1e-6 else 0.0
            self.get_logger().info(
                f"{side} box detected. count={count}. Driving forward {self.box_forward_before_turn_distance:.2f} m "
                f"({fwd_time:.2f} s) along line before turning."
            )
        else:
            if self.start_relative_turn(
                self.side_sign(side) * self.turn_angle_90_deg,
                self.STATE_BOX_TURN_TO_BOX,
                self.STATE_BOX_DRIVE_TO_BOX
            ):
                self.get_logger().info(
                    f"{side} box detected. count={count}. Starting gyro turn toward box."
                )

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def publish_task_done(self):
        if self.task2_done_published:
            return
        msg = String()
        msg.data = 'DONE'
        self.task_status_pub.publish(msg)
        self.task2_done_published = True
        self.get_logger().info("Published Task 2 DONE status.")

    def start_relative_turn(self, delta_deg: float, turn_state: str, next_state: str) -> bool:
        if not self.gyro_ready or self.current_yaw_deg is None:
            self.get_logger().warn("Gyro angle not ready yet. Waiting before turn.")
            self.state = turn_state
            self.turn_active = False
            self.turn_next_state = next_state
            return False

        self.turn_target_deg = self.normalize_angle_deg(self.current_yaw_deg + delta_deg)
        self.turn_next_state = next_state
        self.turn_start_time = self.get_clock().now()
        self.turn_active = True
        self.state = turn_state

        self.get_logger().info(
            f"Starting turn: current={self.current_yaw_deg:.2f} deg, "
            f"delta={delta_deg:.2f} deg, target={self.turn_target_deg:.2f} deg, "
            f"next_state={next_state}"
        )
        return True

    def execute_gyro_turn(self, twist: Twist):
        twist.linear.x = 0.0
        twist.angular.z = 0.0

        if not self.gyro_ready or self.current_yaw_deg is None:
            return

        if not self.turn_active:
            if self.state == self.STATE_LINE_CROSS_TURN:
                self.start_relative_turn(
                    self.line_cross_turn_direction * self.turn_angle_90_deg,
                    self.STATE_LINE_CROSS_TURN,
                    self.STATE_LINE_CROSS_POST_WAIT
                )
            elif self.state == self.STATE_BOX_TURN_TO_BOX:
                self.start_relative_turn(
                    self.side_sign(self.active_box_side) * self.turn_angle_90_deg,
                    self.STATE_BOX_TURN_TO_BOX,
                    self.STATE_BOX_DRIVE_TO_BOX
                )
            elif self.state == self.STATE_BOX_TURN_TO_RESUME:
                self.start_relative_turn(
                    self.opposite_side_sign(self.active_box_side) * self.turn_angle_90_deg,
                    self.STATE_BOX_TURN_TO_RESUME,
                    self.STATE_FOLLOW_LINE
                )
            return

        error_deg = self.angle_diff_deg(self.turn_target_deg, self.current_yaw_deg)

        if abs(error_deg) <= self.turn_tolerance_deg:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.turn_active = False

            if self.state == self.STATE_LINE_CROSS_TURN:
                self.state = self.STATE_LINE_CROSS_POST_WAIT
                self.post_turn_wait_start_time = self.get_clock().now()
                self.get_logger().info(
                    f"Line-cross turn complete at yaw={self.current_yaw_deg:.2f} deg. "
                    f"Waiting {self.post_turn_wait_time:.2f}s."
                )
            else:
                self.state = self.turn_next_state
                if self.state == self.STATE_BOX_DRIVE_TO_BOX:
                    self.box_stop_counter = 0
                    self.box_drive_start_time = self.get_clock().now()
                    self.red_lost_counter = 0
                    self.get_logger().info(
                        f"Turned toward {self.active_box_side} box. Centering red box, stopping by front distance sensor."
                    )
                elif self.state == self.STATE_FOLLOW_LINE:
                    handled_side = self.active_box_side
                    self.box_stop_counter = 0
                    self.red_lost_counter = 0
                    self.box_drive_start_time = None
                    self.reverse_start_time = None
                    self.active_box_side = None
                    if handled_side is not None:
                        self.start_same_box_ignore(handled_side)
                    self.get_logger().info("Resume turn complete. Switching directly to white-line following.")
            return

        elapsed = (self.get_clock().now() - self.turn_start_time).nanoseconds / 1e9
        if elapsed >= self.turn_timeout_sec:
            self.get_logger().warn(
                f"Turn timeout. current={self.current_yaw_deg:.2f} target={self.turn_target_deg:.2f} "
                f"error={error_deg:.2f}. Forcing next state."
            )
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.turn_active = False

            if self.state == self.STATE_LINE_CROSS_TURN:
                self.state = self.STATE_LINE_CROSS_POST_WAIT
                self.post_turn_wait_start_time = self.get_clock().now()
            else:
                self.state = self.turn_next_state
                if self.state == self.STATE_BOX_DRIVE_TO_BOX:
                    self.box_drive_start_time = self.get_clock().now()
                    self.red_lost_counter = 0
                elif self.state == self.STATE_FOLLOW_LINE:
                    handled_side = self.active_box_side
                    self.box_stop_counter = 0
                    self.red_lost_counter = 0
                    self.box_drive_start_time = None
                    self.reverse_start_time = None
                    self.active_box_side = None
                    if handled_side is not None:
                        self.start_same_box_ignore(handled_side)
            return

        speed = self.turn_kp * abs(error_deg)
        speed = self.clamp(speed, self.turn_min_speed, self.turn_max_speed)

        twist.linear.x = 0.0
        twist.angular.z = speed if error_deg > 0.0 else -speed

    def run_line_cross_sequence(self, bottom_area: float, twist: Twist):
        if self.state == self.STATE_LINE_CROSS_APPROACH:
            twist.linear.x = self.line_cross_speed
            twist.angular.z = 0.0

            if bottom_area > self.bottom_min_area:
                self.line_seen = True
                self.line_gone_counter = 0
                self.state = self.STATE_LINE_CROSS_WAIT_DISAPPEAR
                self.get_logger().info("White line reached robot area. Waiting until it disappears...")

        elif self.state == self.STATE_LINE_CROSS_WAIT_DISAPPEAR:
            twist.linear.x = self.line_cross_speed
            twist.angular.z = 0.0

            if bottom_area > self.bottom_min_area:
                self.line_gone_counter = 0
            else:
                self.line_gone_counter += 1

            if self.line_seen and self.line_gone_counter >= self.line_gone_frames:
                self.state = self.STATE_LINE_CROSS_EXTRA_FORWARD
                self.extra_forward_start_time = self.get_clock().now()
                self.get_logger().info(
                    f"White line passed under robot. Moving extra {self.extra_forward_distance:.2f} m."
                )

        elif self.state == self.STATE_LINE_CROSS_EXTRA_FORWARD:
            twist.linear.x = self.line_cross_speed
            twist.angular.z = 0.0

            elapsed = (self.get_clock().now() - self.extra_forward_start_time).nanoseconds / 1e9
            if elapsed >= self.extra_forward_time:
                self.start_relative_turn(
                    self.line_cross_turn_direction * self.turn_angle_90_deg,
                    self.STATE_LINE_CROSS_TURN,
                    self.STATE_LINE_CROSS_POST_WAIT
                )

        elif self.state == self.STATE_LINE_CROSS_TURN:
            self.execute_gyro_turn(twist)

        elif self.state == self.STATE_LINE_CROSS_POST_WAIT:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

            elapsed = (self.get_clock().now() - self.post_turn_wait_start_time).nanoseconds / 1e9
            if elapsed >= self.post_turn_wait_time:
                self.state = self.line_cross_next_state
                if self.state == self.STATE_FOLLOW_LINE:
                    self.start_measurement()
                    if not self.startup_box_ignore_used:
                        self.start_startup_box_ignore()
                self.get_logger().info(f"Line-cross sequence complete. Next state: {self.state}")

    def should_finish_task2(self) -> bool:
        if self.boxes_completed < self.target_box_count:
            self.finish_wall_counter = 0
            return False

        front_hit = (
            self.valid_range(self.front_range)
            and self.front_range <= self.task2_finish_wall_distance
        )

        if front_hit:
            self.finish_wall_counter += 1
        else:
            self.finish_wall_counter = 0

        return self.finish_wall_counter >= self.task2_finish_wall_frames

    def check_pick_server(self):
        if self.pick_client.server_is_ready() and not self.pick_server_ready_logged:
            self.pick_server_ready_logged = True
            self.get_logger().info(f"Pick action server ready: {self.pick_action_name}")

    def send_pick_goal(self):
        if self.active_box_side is None:
            self.get_logger().warn("Cannot send pick goal: active_box_side is None.")
            return

        if self.pick_goal_send_once and self.pick_goal_sent:
            return

        if not self.pick_client.server_is_ready():
            self.get_logger().warn(
                f"Pick action server '{self.pick_action_name}' not available yet."
            )
            self.pick_result_ready = True
            self.pick_result_success = False
            self.pick_result_message = "Action server not available"
            return

        goal_msg = PickBox.Goal()
        goal_msg.side = self.active_box_side

        self.pick_goal_sent = True
        self.pick_in_progress = True
        self.pick_result_ready = False
        self.pick_result_success = False
        self.pick_result_message = ""
        self.pick_feedback_text = "goal_sent"

        self.get_logger().info(f"Sending PickBox goal for side={self.active_box_side}")

        future = self.pick_client.send_goal_async(
            goal_msg,
            feedback_callback=self.pick_feedback_callback
        )
        future.add_done_callback(self.pick_goal_response_callback)

    def pick_feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.pick_feedback_text = f"{fb.current_step} ({fb.progress:.2f})"

    def pick_goal_response_callback(self, future):
        goal_handle = future.result()

        if goal_handle is None:
            self.pick_in_progress = False
            self.pick_result_ready = True
            self.pick_result_success = False
            self.pick_result_message = "No goal handle returned"
            self.get_logger().error("Pick goal failed: no goal handle returned.")
            return

        if not goal_handle.accepted:
            self.pick_in_progress = False
            self.pick_result_ready = True
            self.pick_result_success = False
            self.pick_result_message = "Goal rejected by server"
            self.get_logger().warn("Pick goal was rejected by action server.")
            return

        self.current_goal_handle = goal_handle
        self.get_logger().info("Pick goal accepted by action server.")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.pick_result_callback)

    def pick_result_callback(self, future):
        self.pick_in_progress = False

        try:
            result_wrap = future.result()
            result = result_wrap.result
        except Exception as e:
            self.pick_result_ready = True
            self.pick_result_success = False
            self.pick_result_message = f"Exception while getting result: {e}"
            self.get_logger().error(self.pick_result_message)
            return

        self.pick_result_ready = True
        self.pick_result_success = bool(result.success)
        self.pick_result_message = str(result.message)

        if self.pick_result_success:
            self.get_logger().info(f"Pick action success: {self.pick_result_message}")
        else:
            self.get_logger().warn(f"Pick action failed: {self.pick_result_message}")

    def _draw_tof_sparkline(self):
        """
        Draw left (cyan) and right (yellow) ToF distance sparklines in a
        separate window. Range 0–2 m maps to the plot height. A horizontal
        line shows the detection threshold drop from each baseline.
        """
        if not self.debug:
            return
        plot_h = 150
        plot_w = 400
        max_m = 2.0

        canvas = np.zeros((plot_h + 30, plot_w, 3), dtype=np.uint8)

        def draw_line(buf, color):
            pts = list(buf)
            if len(pts) < 2:
                return
            step = plot_w / (len(pts) - 1)
            for i in range(len(pts) - 1):
                v1 = min(pts[i], max_m) / max_m
                v2 = min(pts[i + 1], max_m) / max_m
                px1 = int(i * step)
                py1 = int(plot_h - v1 * plot_h)
                px2 = int((i + 1) * step)
                py2 = int(plot_h - v2 * plot_h)
                cv2.line(canvas, (px1, py1), (px2, py2), color, 1)

        draw_line(self.left_vis_buf, (255, 255, 0))
        draw_line(self.right_vis_buf, (0, 255, 255))

        # threshold line for each side (baseline - delta_threshold)
        for buf, color in [(self.left_baseline_buf, (255, 255, 0)),
                           (self.right_baseline_buf, (0, 255, 255))]:
            if len(buf) >= 3:
                thresh = float(np.median(list(buf))) - self.tof_delta_threshold
                thresh = max(0.0, min(thresh, max_m))
                ty = int(plot_h - (thresh / max_m) * plot_h)
                cv2.line(canvas, (0, ty), (plot_w, ty), color, 1)

        cv2.putText(canvas, f"L={self.fmt_range(self.left_range)}  R={self.fmt_range(self.right_range)}  thresh_drop={self.tof_delta_threshold:.2f}m",
                    (5, plot_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("ToF distances", canvas)

    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]

        y0 = int(h * self.roi_y_start)
        roi = frame[y0:h, 0:w]
        mask = build_white_mask(roi, self.h_low, self.s_low, self.v_low, self.h_high, self.s_high, self.v_high)

        M = cv2.moments(mask)
        area = M["m00"]

        bh = int(h * self.bottom_strip_height_ratio)
        by0 = max(0, h - bh)
        bottom_roi = frame[by0:h, 0:w]
        bottom_mask = build_white_mask(bottom_roi, self.h_low, self.s_low, self.v_low, self.h_high, self.s_high, self.v_high)
        Mb = cv2.moments(bottom_mask)
        bottom_area = Mb["m00"]

        # TODO: re-enable after turning is implemented
        # red_found, red_cx, red_cy, red_area, red_bbox, red_mask = self.detect_red_box(frame)
        red_found, red_cx, red_cy, red_area, red_bbox, red_mask = False, 0, 0, 0.0, None, None

        twist = Twist()

        if self.state in {
            self.STATE_LINE_CROSS_APPROACH,
            self.STATE_LINE_CROSS_WAIT_DISAPPEAR,
            self.STATE_LINE_CROSS_EXTRA_FORWARD,
            self.STATE_LINE_CROSS_TURN,
            self.STATE_LINE_CROSS_POST_WAIT,
        }:
            self.run_line_cross_sequence(bottom_area, twist)

        elif self.state == self.STATE_FOLLOW_LINE:
            if self.front_obstacle_should_stop():
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.state = self.STATE_TASK2_DONE
                self.task2_done = True
                self.stop_robot()
                self.publish_task_done()
                self.get_logger().info(
                    f"Front obstacle/wall detected at {self.fmt_range(self.front_range)} m. Ending task."
                )

            elif self.boxes_completed >= self.target_box_count:
                twist.linear.x = self.task2_finish_forward_speed
                twist.angular.z = 0.0

                if self.should_finish_task2():
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.state = self.STATE_TASK2_DONE
                    self.task2_done = True
                    self.stop_robot()
                    self.publish_task_done()
                    self.get_logger().info(
                        f"Task 2 complete: boxes_completed={self.boxes_completed}, "
                        f"front wall reached. front_range={self.fmt_range(self.front_range)}"
                    )
            else:
                if area > self.min_area:
                    cx = int(M["m10"] / area)
                    error = float(cx - (w // 2))
                    ang = -self.kp * error
                    ang = self.clamp(ang, -self.max_angular, self.max_angular)

                    twist.linear.x = self.linear_speed
                    twist.angular.z = ang
                else:
                    twist.linear.x = self.search_linear
                    twist.angular.z = self.search_angular

                side = self.choose_box_side()
                if side is not None:
                    # Stop the robot and blink red LED twice to signal detection.
                    # _start_blink sets STATE_BOX_BLINK; after the blink sequence
                    # the node returns here automatically via _blink_step().
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self._start_blink(side)

        elif self.state == self.STATE_BOX_BLINK:
            # Robot stays stopped while the LED blink timer runs in the background.
            # _blink_step() handles the state transition back to STATE_FOLLOW_LINE.
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        elif self.state == self.STATE_BOX_FORWARD_BEFORE_TURN:
            # Drive forward along the white line for the configured distance
            # before turning 90° to approach the box
            if area > self.min_area:
                cx = int(M["m10"] / area)
                error = float(cx - (w // 2))
                ang = -self.kp * error
                ang = self.clamp(ang, -self.max_angular, self.max_angular)
                twist.linear.x = self.linear_speed
                twist.angular.z = ang
            else:
                twist.linear.x = self.linear_speed
                twist.angular.z = 0.0

            fwd_time = (
                self.box_forward_before_turn_distance / self.linear_speed
                if self.linear_speed > 1e-6 else 0.0
            )
            elapsed = (self.get_clock().now() - self.box_forward_before_turn_start_time).nanoseconds / 1e9
            if elapsed >= fwd_time:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                if self.start_relative_turn(
                    self.side_sign(self.active_box_side) * self.turn_angle_90_deg,
                    self.STATE_BOX_TURN_TO_BOX,
                    self.STATE_BOX_DRIVE_TO_BOX
                ):
                    self.get_logger().info(
                        f"Forward {self.box_forward_before_turn_distance:.2f} m complete. "
                        f"Starting gyro turn toward {self.active_box_side} box."
                    )

        elif self.state == self.STATE_BOX_TURN_TO_BOX:
            self.execute_gyro_turn(twist)

        elif self.state == self.STATE_BOX_DRIVE_TO_BOX:
            side_dist = self.current_side_range()
            front_dist = self.front_range
            near_by_range = self.valid_range(front_dist) and front_dist <= self.box_front_stop_distance

            # TODO: re-enable red-box visual tracking after turning is implemented
            # if red_found:
            #     self.red_lost_counter = 0
            #     error = float(red_cx - (w // 2))
            #     ang = -self.red_kp * error
            #     ang = self.clamp(ang, -self.red_max_angular, self.red_max_angular)
            #     twist.linear.x = self.box_approach_speed
            #     twist.angular.z = ang
            # else:
            #     self.red_lost_counter += 1
            #     twist.linear.x = 0.0
            #     twist.angular.z = self.side_sign(self.active_box_side) * self.red_search_angular
            #     if self.red_lost_counter > self.red_lost_frames_limit:
            #         self.get_logger().info("Red box lost. Rotating slowly to reacquire target.")
            twist.linear.x = self.box_approach_speed
            twist.angular.z = 0.0

            if near_by_range:
                self.box_stop_counter += 1
            else:
                self.box_stop_counter = 0

            if self.box_stop_counter >= self.box_stop_frames:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.state = self.STATE_BOX_REQUEST_PICK
                self.get_logger().info(
                    f"Reached {self.active_box_side} box using front distance stop. "
                    f"front={self.fmt_range(front_dist)} side={self.fmt_range(side_dist)}. "
                    f"Stopping and requesting pick."
                )
            elif self.box_drive_start_time is not None:
                elapsed = (self.get_clock().now() - self.box_drive_start_time).nanoseconds / 1e9
                if elapsed >= self.box_drive_timeout_sec:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.state = self.STATE_BOX_REQUEST_PICK
                    self.get_logger().warn(
                        f"Box drive timeout. Proceeding to pick request anyway. "
                        f"front={self.fmt_range(front_dist)} side={self.fmt_range(side_dist)} "
                        f"active_box={self.active_box_side}"
                    )

        elif self.state == self.STATE_BOX_REQUEST_PICK:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

            self.send_pick_goal()
            self.state = self.STATE_BOX_WAIT_PICK_RESULT
            self.get_logger().info("Pick request sent. Waiting for result.")

        elif self.state == self.STATE_BOX_WAIT_PICK_RESULT:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

            if self.pick_result_ready:
                if self.pick_result_success:
                    self.boxes_completed += 1
                    self.finish_wall_counter = 0
                    self.state = self.STATE_BOX_REVERSE_AFTER_PICK
                    self.reverse_start_time = self.get_clock().now()
                    self.get_logger().info(
                        f"Pick success. boxes_completed={self.boxes_completed}. Reversing away from box."
                    )
                else:
                    self.state = self.STATE_BOX_PICK_FAILED
                    self.get_logger().warn(
                        f"Pick failed. message={self.pick_result_message}"
                    )

        elif self.state == self.STATE_BOX_REVERSE_AFTER_PICK:
            twist.linear.x = -self.reverse_after_pick_speed
            twist.angular.z = 0.0

            elapsed = (self.get_clock().now() - self.reverse_start_time).nanoseconds / 1e9
            if elapsed >= self.reverse_after_pick_time:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.start_relative_turn(
                    self.opposite_side_sign(self.active_box_side) * self.turn_angle_90_deg,
                    self.STATE_BOX_TURN_TO_RESUME,
                    self.STATE_FOLLOW_LINE
                )
                self.get_logger().info(
                    f"Reverse complete. Turning opposite of previous detour for side={self.active_box_side}."
                )

        elif self.state == self.STATE_BOX_TURN_TO_RESUME:
            self.execute_gyro_turn(twist)

        elif self.state == self.STATE_BOX_PICK_FAILED:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

            if self.pick_retry_count < self.pick_retry_limit:
                self.pick_retry_count += 1
                self.pick_goal_sent = False
                self.pick_in_progress = False
                self.pick_result_ready = False
                self.pick_result_success = False
                self.pick_result_message = ""
                self.pick_feedback_text = ""
                self.state = self.STATE_BOX_REQUEST_PICK
                self.get_logger().warn(
                    f"Retrying pick action: attempt {self.pick_retry_count}/{self.pick_retry_limit}"
                )
            else:
                self.get_logger().warn(
                    f"Pick failed and retry limit ({self.pick_retry_limit}) reached. "
                    f"Skipping box on {self.active_box_side} side. Reversing to resume line."
                )
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.state = self.STATE_BOX_REVERSE_AFTER_PICK
                self.reverse_start_time = self.get_clock().now()

        elif self.state == self.STATE_TASK2_DONE:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

        if self.print_distances_every_frame and self.measurement_started:
            ignore_side = self.ignore_box_side if self.ignore_box_side is not None else "None"
            startup_ignore = self.startup_ignore_active_now()
            self.get_logger().info(
                f"STATE={self.state} "
                f"yaw={self.current_yaw_deg if self.current_yaw_deg is not None else 'None'} "
                f"turn_target={self.turn_target_deg if self.turn_target_deg is not None else 'None'} "
                f"left_raw={self.fmt_range(self.left_range_raw)} left_f={self.fmt_range(self.left_range)} "
                f"right_raw={self.fmt_range(self.right_range_raw)} right_f={self.fmt_range(self.right_range)} "
                f"front_raw={self.fmt_range(self.front_range_raw)} front_f={self.fmt_range(self.front_range)} "
                f"startup_ignore={startup_ignore} ignore_side={ignore_side} "
                f"front_stop_counter={self.front_obstacle_counter}/{self.front_obstacle_stop_frames} "
                f"L_det={self.left_detect_counter}/{self.tof_confirm_frames} "
                f"R_det={self.right_detect_counter}/{self.tof_confirm_frames} "
                f"pick_in_progress={self.pick_in_progress} pick_feedback='{self.pick_feedback_text}' "
                f"boxes_completed={self.boxes_completed} "
                f"finish_counter={self.finish_wall_counter}/{self.task2_finish_wall_frames} "
                f"cmd(v={twist.linear.x:.2f}, w={twist.angular.z:.2f})"
            )

        vis = frame.copy()

        cv2.rectangle(vis, (0, y0), (w - 1, h - 1), (0, 255, 0), 2)
        cv2.putText(vis, "FOLLOW ROI", (10, max(25, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.rectangle(vis, (0, by0), (w - 1, h - 1), (255, 0, 0), 2)
        cv2.putText(vis, "BOTTOM CHECK", (10, max(50, by0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if area > self.min_area:
            cx_vis = int(M["m10"] / area)
            cy_vis = y0 + (h - y0) // 2
            cv2.circle(vis, (cx_vis, cy_vis), 8, (0, 255, 255), -1)
            cv2.line(vis, (w // 2, y0), (w // 2, h - 1), (255, 255, 0), 2)

        # TODO: re-enable red box visualization after turning is implemented
        # if red_found and red_bbox is not None:
        #     x, y, bw, bh = red_bbox
        #     cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        #     cv2.circle(vis, (red_cx, red_cy), 6, (0, 0, 255), -1)
        #     cv2.line(vis, (w // 2, 0), (w // 2, h - 1), (0, 0, 255), 1)
        #     cv2.putText(vis, f"red_area={int(red_area)}", (x, max(20, y - 10)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(vis, f"STATE: {self.state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(vis, f"main_area={int(area)} bottom_area={int(bottom_area)}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(
            vis,
            f"L={self.display_range(self.left_range)} R={self.display_range(self.right_range)} F={self.display_range(self.front_range)}",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            vis,
            f"Lraw={self.fmt_range(self.left_range_raw)} Rraw={self.fmt_range(self.right_range_raw)} Fraw={self.fmt_range(self.front_range_raw)}",
            (10, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 255),
            2
        )

        ignore_txt = self.ignore_box_side if self.ignore_box_side is not None else "None"
        startup_ignore_txt = "ON" if self.startup_ignore_active_now() else "OFF"
        cv2.putText(
            vis,
            f"startup_ignore={startup_ignore_txt} ignore_side={ignore_txt}",
            (10, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 140),
            2
        )

        # Show delta-detection state: baseline, current reading, confirm counter
        l_base_txt = (
            f"{float(np.median(list(self.left_baseline_buf))):.2f}"
            if len(self.left_baseline_buf) >= 3 else "priming"
        )
        r_base_txt = (
            f"{float(np.median(list(self.right_baseline_buf))):.2f}"
            if len(self.right_baseline_buf) >= 3 else "priming"
        )
        cv2.putText(
            vis,
            f"Lbase={l_base_txt} Rbase={r_base_txt} dThresh={self.tof_delta_threshold:.2f} "
            f"Ldet={self.left_detect_counter}/{self.tof_confirm_frames} "
            f"Rdet={self.right_detect_counter}/{self.tof_confirm_frames}",
            (10, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 210, 120),
            1
        )

        cv2.putText(
            vis,
            f"front_stop={self.front_obstacle_counter}/{self.front_obstacle_stop_frames}",
            (10, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 220, 150),
            2
        )

        cv2.putText(
            vis,
            f"LEFT count={self.left_box_count} RIGHT count={self.right_box_count}",
            (10, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2
        )

        cv2.putText(
            vis,
            f"boxes_completed={self.boxes_completed}/{self.target_box_count}",
            (10, 275),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 220, 120),
            2
        )

        cv2.putText(
            vis,
            f"active_box={self.active_box_side}",
            (10, 305),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 255, 200),
            2
        )

        yaw_txt = "None" if self.current_yaw_deg is None else f"{self.current_yaw_deg:.1f}"
        tgt_txt = "None" if self.turn_target_deg is None else f"{self.turn_target_deg:.1f}"
        cv2.putText(
            vis,
            f"yaw={yaw_txt} target={tgt_txt}",
            (10, 335),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 255, 255),
            2
        )

        cv2.putText(
            vis,
            f"cmd v={twist.linear.x:.2f} w={twist.angular.z:.2f}",
            (10, 365),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        if self.debug:
            self._draw_tof_sparkline()
            cv2.imshow("camera", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("Quit requested. Stopping robot.")
                self.stop_robot()
                rclpy.shutdown()
                cv2.destroyAllWindows()
                return

        self.frame_count += 1
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds > 1_000_000_000:
            self.last_log_time = now
            self.get_logger().info(
                f"fps~{self.frame_count} state={self.state} "
                f"yaw={yaw_txt} target={tgt_txt} "
                f"main_area={int(area)} bottom_area={int(bottom_area)} "
                f"left={self.display_range(self.left_range)} "
                f"right={self.display_range(self.right_range)} "
                f"front={self.display_range(self.front_range)} "
                f"startup_ignore={startup_ignore_txt} "
                f"ignore_side={ignore_txt} "
                f"counts(L,R)=({self.left_box_count},{self.right_box_count}) "
                f"boxes_completed={self.boxes_completed}/{self.target_box_count} "
                f"active_box={self.active_box_side} "
                f"pick_in_progress={self.pick_in_progress} "
                f"pick_result_ready={self.pick_result_ready} "
                f"pick_feedback='{self.pick_feedback_text}' "
                f"cmd(v,w)=({twist.linear.x:.2f},{twist.angular.z:.2f})"
            )
            self.frame_count = 0
        
def main():
    rclpy.init()
    node = WhiteLineFollowerWithBoxVisit()
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