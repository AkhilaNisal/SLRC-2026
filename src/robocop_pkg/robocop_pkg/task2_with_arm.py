#!/usr/bin/env python3

# starting pos in arena :translation 0.75 -1.05 0

import math
from collections import deque

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

from robot_arm_interfaces.action import PickBox


class SideBoxVariationDetector:
    def __init__(
        self,
        name: str,
        fast_alpha: float,
        baseline_alpha: float,
        start_drop: float,
        peak_drop: float,
        end_drop: float,
        freeze_drop: float,
        min_duration_sec: float,
        max_duration_sec: float,
        min_area: float,
        recovery_samples: int,
        cooldown_sec: float,
        history_len: int,
    ):
        self.name = name
        self.fast_alpha = float(fast_alpha)
        self.baseline_alpha = float(baseline_alpha)
        self.start_drop = float(start_drop)
        self.peak_drop = float(peak_drop)
        self.end_drop = float(end_drop)
        self.freeze_drop = float(freeze_drop)
        self.min_duration_sec = float(min_duration_sec)
        self.max_duration_sec = float(max_duration_sec)
        self.min_area = float(min_area)
        self.recovery_samples = int(recovery_samples)
        self.cooldown_sec = float(cooldown_sec)
        self.history_len = int(history_len)

        self.raw_hist = deque(maxlen=self.history_len)
        self.fast_hist = deque(maxlen=self.history_len)
        self.baseline_hist = deque(maxlen=self.history_len)
        self.excursion_hist = deque(maxlen=self.history_len)

        self.reset()

    def reset(self):
        self.fast = None
        self.baseline = None
        self.prev_fast = None
        self.prev_time_sec = None

        self.tracking = False
        self.event_sent_in_current_valley = False
        self.valley_start_sec = None
        self.peak_excursion = 0.0
        self.excursion_area = 0.0
        self.recovery_count = 0
        self.cooldown_until_sec = 0.0
        self.last_event_sec = -1e9

        self.last_info = {
            'name': self.name,
            'raw': math.nan,
            'fast': math.nan,
            'baseline': math.nan,
            'excursion': 0.0,
            'slope': 0.0,
            'tracking': False,
            'peak_excursion': 0.0,
            'excursion_area': 0.0,
            'recovery_count': 0,
            'duration': 0.0,
            'cooldown_left': 0.0,
            'event': False,
        }

        self.raw_hist.clear()
        self.fast_hist.clear()
        self.baseline_hist.clear()
        self.excursion_hist.clear()

    def _append_history(self, raw_value, fast_value, baseline_value, excursion_value):
        self.raw_hist.append(raw_value)
        self.fast_hist.append(fast_value)
        self.baseline_hist.append(baseline_value)
        self.excursion_hist.append(excursion_value)

    def _reset_valley_only(self):
        self.tracking = False
        self.event_sent_in_current_valley = False
        self.valley_start_sec = None
        self.peak_excursion = 0.0
        self.excursion_area = 0.0
        self.recovery_count = 0

    def update(self, raw_value: float, now_sec: float, enabled: bool = True):
        event = False

        valid = (
            raw_value is not None
            and not math.isinf(raw_value)
            and not math.isnan(raw_value)
            and raw_value > 0.0
        )

        if not valid:
            self._append_history(math.nan, math.nan, math.nan, math.nan)
            self._reset_valley_only()
            self.prev_time_sec = now_sec
            self.last_info = {
                **self.last_info,
                'raw': math.nan,
                'fast': math.nan,
                'baseline': math.nan,
                'excursion': 0.0,
                'slope': 0.0,
                'tracking': False,
                'peak_excursion': 0.0,
                'excursion_area': 0.0,
                'recovery_count': 0,
                'duration': 0.0,
                'cooldown_left': max(0.0, self.cooldown_until_sec - now_sec),
                'event': False,
            }
            return False, self.last_info

        if self.fast is None or self.baseline is None:
            self.fast = float(raw_value)
            self.baseline = float(raw_value)
            self.prev_fast = float(raw_value)
            self.prev_time_sec = now_sec
            self._append_history(float(raw_value), self.fast, self.baseline, 0.0)
            self.last_info = {
                **self.last_info,
                'raw': float(raw_value),
                'fast': self.fast,
                'baseline': self.baseline,
                'excursion': 0.0,
                'slope': 0.0,
                'tracking': False,
                'peak_excursion': 0.0,
                'excursion_area': 0.0,
                'recovery_count': 0,
                'duration': 0.0,
                'cooldown_left': max(0.0, self.cooldown_until_sec - now_sec),
                'event': False,
            }
            return False, self.last_info

        dt = 0.0
        if self.prev_time_sec is not None:
            dt = max(0.0, min(0.25, now_sec - self.prev_time_sec))

        prev_fast = self.fast
        self.fast = self.fast_alpha * float(raw_value) + (1.0 - self.fast_alpha) * self.fast
        slope = self.fast - prev_fast

        raw_excursion = max(0.0, self.baseline - self.fast)

        if (not self.tracking and raw_excursion < self.freeze_drop) or (not enabled):
            self.baseline = (
                self.baseline_alpha * self.fast +
                (1.0 - self.baseline_alpha) * self.baseline
            )

        excursion = max(0.0, self.baseline - self.fast)

        if not enabled:
            self._append_history(float(raw_value), self.fast, self.baseline, excursion)
            self._reset_valley_only()
            self.prev_fast = self.fast
            self.prev_time_sec = now_sec
            self.last_info = {
                **self.last_info,
                'raw': float(raw_value),
                'fast': self.fast,
                'baseline': self.baseline,
                'excursion': excursion,
                'slope': slope,
                'tracking': False,
                'peak_excursion': 0.0,
                'excursion_area': 0.0,
                'recovery_count': 0,
                'duration': 0.0,
                'cooldown_left': max(0.0, self.cooldown_until_sec - now_sec),
                'event': False,
            }
            return False, self.last_info

        if not self.tracking:
            if now_sec >= self.cooldown_until_sec and excursion >= self.start_drop and slope <= 0.0:
                self.tracking = True
                self.event_sent_in_current_valley = False
                self.valley_start_sec = now_sec
                self.peak_excursion = excursion
                self.excursion_area = 0.0
                self.recovery_count = 0
        else:
            duration = max(0.0, now_sec - self.valley_start_sec)
            self.peak_excursion = max(self.peak_excursion, excursion)
            self.excursion_area += excursion * dt

            if slope > 0.0:
                self.recovery_count += 1
            elif slope < 0.0:
                self.recovery_count = 0

            if (
                not self.event_sent_in_current_valley
                and duration >= self.min_duration_sec
                and self.peak_excursion >= self.peak_drop
                and self.excursion_area >= self.min_area
                and self.recovery_count >= self.recovery_samples
            ):
                event = True
                self.event_sent_in_current_valley = True
                self.cooldown_until_sec = now_sec + self.cooldown_sec
                self.last_event_sec = now_sec

            if excursion <= self.end_drop or duration >= self.max_duration_sec:
                self._reset_valley_only()

        duration = 0.0
        if self.tracking and self.valley_start_sec is not None:
            duration = max(0.0, now_sec - self.valley_start_sec)

        self._append_history(float(raw_value), self.fast, self.baseline, excursion)
        self.prev_fast = self.fast
        self.prev_time_sec = now_sec
        self.last_info = {
            'name': self.name,
            'raw': float(raw_value),
            'fast': self.fast,
            'baseline': self.baseline,
            'excursion': excursion,
            'slope': slope,
            'tracking': self.tracking,
            'peak_excursion': self.peak_excursion,
            'excursion_area': self.excursion_area,
            'recovery_count': self.recovery_count,
            'duration': duration,
            'cooldown_left': max(0.0, self.cooldown_until_sec - now_sec),
            'event': event,
        }
        return event, self.last_info


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
        self.declare_parameter('forward_speed', 0.1)
        self.declare_parameter('linear_speed', 0.1)
        self.declare_parameter('kp', 0.004)
        self.declare_parameter('max_angular', 1.2)

        self.declare_parameter('extra_forward_distance', 0.28)
        self.declare_parameter('post_turn_wait_time', 1.0)

        self.declare_parameter('search_linear', 0.04)
        self.declare_parameter('search_angular', 0.35)

        # front wall approach after leaving line follower
        self.declare_parameter('front_wall_detect_distance', 0.40)
        self.declare_parameter('front_wall_stop_distance', 0.20)
        self.declare_parameter('front_wall_approach_speed', 0.08)

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
        self.declare_parameter('roi_y_start', 0.60)
        self.declare_parameter('min_area', 5000)

        self.declare_parameter('bottom_strip_height_ratio', 0.14)
        self.declare_parameter('bottom_min_area', 2500)
        self.declare_parameter('line_gone_frames', 5)

        # self.declare_parameter('h_low', 0)
        # self.declare_parameter('s_low', 0)
        # self.declare_parameter('v_low', 180)
        # self.declare_parameter('h_high', 180)
        # self.declare_parameter('s_high', 70)
        # self.declare_parameter('v_high', 255)


        # Black line detection (for black line on white floor)
        # =========================
        self.declare_parameter('h_low', 0)
        self.declare_parameter('s_low', 0)
        self.declare_parameter('v_low', 0)        # Dark pixels
        self.declare_parameter('h_high', 180)
        self.declare_parameter('s_high', 255)
        self.declare_parameter('v_high', 100)     # Low brightness

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
        self.declare_parameter('red_roi_top_trim_ratio', 0.18)
        self.declare_parameter('red_roi_side_trim_ratio', 0.12)
        self.declare_parameter('red_confirm_timeout_sec', 0.70)
        self.declare_parameter('red_confirm_frames', 2)

        # =========================
        # Distance sensing / filter
        # =========================
        self.declare_parameter('range_filter_alpha', 0.1)
        self.declare_parameter('left_range_filter_alpha', 0.3)
        self.declare_parameter('print_distances_every_frame', True)

        # =========================
        # Box detection while following line
        # =========================
        self.declare_parameter('left_box_detect_distance', 0.54)
        self.declare_parameter('right_box_detect_distance', 0.45)
        self.declare_parameter('box_detect_frames', 10)

        self.declare_parameter('startup_box_ignore_distance', 0.30)
        self.declare_parameter('same_box_ignore_distance', 0.35)

        # box detection by side-sensor variation pattern (relative valley detection)
        self.declare_parameter('show_debug_windows', True)
        self.declare_parameter('side_history_length', 220)
        self.declare_parameter('side_variation_fast_alpha', 0.35)
        self.declare_parameter('side_variation_baseline_alpha', 0.03)
        self.declare_parameter('side_variation_start_drop_m', 0.022)
        self.declare_parameter('side_variation_peak_drop_m', 0.040)
        self.declare_parameter('side_variation_end_drop_m', 0.010)
        self.declare_parameter('side_variation_freeze_drop_m', 0.012)
        self.declare_parameter('side_variation_min_duration_sec', 0.35)
        self.declare_parameter('side_variation_max_duration_sec', 4.50)
        self.declare_parameter('side_variation_min_area', 0.040)
        self.declare_parameter('side_variation_recovery_samples', 6)
        self.declare_parameter('side_variation_cooldown_sec', 1.80)

        # front obstacle / wall stop
        self.declare_parameter('front_obstacle_stop_distance', 0.30)
        self.declare_parameter('front_obstacle_stop_frames', 3)

        # =========================
        # Box visit maneuver
        # =========================
        self.declare_parameter('box_forward_before_turn_distance', 0.07)

        # New slower-front-stop approach settings
        self.declare_parameter('box_slow_distance_m', 0.30)
        self.declare_parameter('box_stop_distance_m', 0.20)
        self.declare_parameter('box_approach_max_speed', 0.10)
        self.declare_parameter('box_stop_settle_sec', 1.0)

        # Old parameters kept for compatibility
        self.declare_parameter('box_approach_speed', 0.08)
        self.declare_parameter('box_stop_distance', 0.15)
        self.declare_parameter('box_front_stop_distance', 0.25)
        self.declare_parameter('box_stop_frames', 1)
        self.declare_parameter('box_return_speed', 0.12)
        self.declare_parameter('box_drive_timeout_sec', 8.0)

        # reverse after pick
        self.declare_parameter('reverse_after_pick_speed', 0.15)
        self.declare_parameter('reverse_after_pick_distance', 0.50)

        # =========================
        # Action client / pickup behavior
        # =========================
        self.declare_parameter('pick_action_name', '/pick_box')
        self.declare_parameter('pick_retry_limit', 5)
        self.declare_parameter('pick_retry_wait_sec', 1.0)
        self.declare_parameter('pick_goal_send_once', True)

        # =========================
        # Task 2 finish behavior
        # =========================
        self.declare_parameter('target_box_count', 6)
        self.declare_parameter('task2_finish_wall_distance', 0.6)
        self.declare_parameter('task2_finish_wall_frames', 10)
        self.declare_parameter('task2_finish_forward_speed', 0.00)

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

        self.front_wall_detect_distance = float(self.get_parameter('front_wall_detect_distance').value)
        self.front_wall_stop_distance = float(self.get_parameter('front_wall_stop_distance').value)
        self.front_wall_approach_speed = float(self.get_parameter('front_wall_approach_speed').value)

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
        self.red_roi_top_trim_ratio = float(self.get_parameter('red_roi_top_trim_ratio').value)
        self.red_roi_side_trim_ratio = float(self.get_parameter('red_roi_side_trim_ratio').value)
        self.red_confirm_timeout_sec = float(self.get_parameter('red_confirm_timeout_sec').value)
        self.red_confirm_frames = int(self.get_parameter('red_confirm_frames').value)

        self.range_filter_alpha = float(self.get_parameter('range_filter_alpha').value)
        self.left_range_filter_alpha = float(self.get_parameter('left_range_filter_alpha').value)
        self.print_distances_every_frame = bool(self.get_parameter('print_distances_every_frame').value)

        self.left_box_detect_distance = float(self.get_parameter('left_box_detect_distance').value)
        self.right_box_detect_distance = float(self.get_parameter('right_box_detect_distance').value)
        self.box_detect_frames = int(self.get_parameter('box_detect_frames').value)

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

        self.show_debug_windows = bool(self.get_parameter('show_debug_windows').value)
        self.side_history_length = int(self.get_parameter('side_history_length').value)
        self.side_variation_fast_alpha = float(self.get_parameter('side_variation_fast_alpha').value)
        self.side_variation_baseline_alpha = float(self.get_parameter('side_variation_baseline_alpha').value)
        self.side_variation_start_drop_m = float(self.get_parameter('side_variation_start_drop_m').value)
        self.side_variation_peak_drop_m = float(self.get_parameter('side_variation_peak_drop_m').value)
        self.side_variation_end_drop_m = float(self.get_parameter('side_variation_end_drop_m').value)
        self.side_variation_freeze_drop_m = float(self.get_parameter('side_variation_freeze_drop_m').value)
        self.side_variation_min_duration_sec = float(self.get_parameter('side_variation_min_duration_sec').value)
        self.side_variation_max_duration_sec = float(self.get_parameter('side_variation_max_duration_sec').value)
        self.side_variation_min_area = float(self.get_parameter('side_variation_min_area').value)
        self.side_variation_recovery_samples = int(self.get_parameter('side_variation_recovery_samples').value)
        self.side_variation_cooldown_sec = float(self.get_parameter('side_variation_cooldown_sec').value)

        self.front_obstacle_stop_distance = float(self.get_parameter('front_obstacle_stop_distance').value)
        self.front_obstacle_stop_frames = int(self.get_parameter('front_obstacle_stop_frames').value)

        self.box_forward_before_turn_distance = float(self.get_parameter('box_forward_before_turn_distance').value)

        self.box_slow_distance_m = float(self.get_parameter('box_slow_distance_m').value)
        self.box_stop_distance_m = float(self.get_parameter('box_stop_distance_m').value)
        self.box_approach_max_speed = float(self.get_parameter('box_approach_max_speed').value)
        self.box_stop_settle_sec = float(self.get_parameter('box_stop_settle_sec').value)

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
        self.pick_retry_wait_sec = float(self.get_parameter('pick_retry_wait_sec').value)
        self.pick_goal_send_once = bool(self.get_parameter('pick_goal_send_once').value)

        self.target_box_count = int(self.get_parameter('target_box_count').value)
        self.task2_finish_wall_distance = float(self.get_parameter('task2_finish_wall_distance').value)
        self.task2_finish_wall_frames = int(self.get_parameter('task2_finish_wall_frames').value)
        self.task2_finish_forward_speed = float(self.get_parameter('task2_finish_forward_speed').value)

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
        self.STATE_FRONT_APPROACH = 'FRONT_APPROACH'

        self.STATE_BOX_FORWARD_BEFORE_TURN = 'BOX_FORWARD_BEFORE_TURN'
        self.STATE_BOX_TURN_TO_BOX = 'BOX_TURN_TO_BOX'
        self.STATE_BOX_DRIVE_TO_BOX = 'BOX_DRIVE_TO_BOX'
        self.STATE_BOX_STOP_SETTLE = 'BOX_STOP_SETTLE'
        self.STATE_BOX_REQUEST_PICK = 'BOX_REQUEST_PICK'
        self.STATE_BOX_WAIT_PICK_RESULT = 'BOX_WAIT_PICK_RESULT'
        self.STATE_BOX_PICK_FAILED = 'BOX_PICK_FAILED'
        self.STATE_BOX_PICK_RETRY_WAIT = 'BOX_PICK_RETRY_WAIT'
        self.STATE_BOX_REVERSE_AFTER_PICK = 'BOX_REVERSE_AFTER_PICK'
        self.STATE_BOX_TURN_TO_RESUME = 'BOX_TURN_TO_RESUME'
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
        self.box_stop_reached_time = None
        self.pick_retry_due_time = None
        self.box_turn_completed_time = None
        self.box_red_confirmed = False
        self.red_confirm_counter = 0
        self.box_turn_completed_time = None
        self.box_red_confirmed = False
        self.red_confirm_counter = 0

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
        self.finish_wall_counter = 0
        self.task2_done = False
        self.task2_done_published = False

        self.left_side_detector = SideBoxVariationDetector(
            name='LEFT',
            fast_alpha=self.side_variation_fast_alpha,
            baseline_alpha=self.side_variation_baseline_alpha,
            start_drop=self.side_variation_start_drop_m,
            peak_drop=self.side_variation_peak_drop_m,
            end_drop=self.side_variation_end_drop_m,
            freeze_drop=self.side_variation_freeze_drop_m,
            min_duration_sec=self.side_variation_min_duration_sec,
            max_duration_sec=self.side_variation_max_duration_sec,
            min_area=self.side_variation_min_area,
            recovery_samples=self.side_variation_recovery_samples,
            cooldown_sec=self.side_variation_cooldown_sec,
            history_len=self.side_history_length,
        )
        self.right_side_detector = SideBoxVariationDetector(
            name='RIGHT',
            fast_alpha=self.side_variation_fast_alpha,
            baseline_alpha=self.side_variation_baseline_alpha,
            start_drop=self.side_variation_start_drop_m,
            peak_drop=self.side_variation_peak_drop_m,
            end_drop=self.side_variation_end_drop_m,
            freeze_drop=self.side_variation_freeze_drop_m,
            min_duration_sec=self.side_variation_min_duration_sec,
            max_duration_sec=self.side_variation_max_duration_sec,
            min_area=self.side_variation_min_area,
            recovery_samples=self.side_variation_recovery_samples,
            cooldown_sec=self.side_variation_cooldown_sec,
            history_len=self.side_history_length,
        )
        self.left_side_info = self.left_side_detector.last_info
        self.right_side_info = self.right_side_detector.last_info
        self.imshow_failed = False
        self.last_red_roi = None

        # debug
        self.frame_count = 0
        self.last_log_time = self.get_clock().now()

        self.configure_line_cross_sequence(
            speed=self.forward_speed,
            turn_direction=+1.0,
            next_state=self.STATE_FOLLOW_LINE
        )

        self.get_logger().info(
            "Started task2_with_arm with front wall exit from white-line following, "
            "front approach to 0.20m stop, slower front-stop box approach, "
            "pick settle wait, action retry wait, same-box ignore and front obstacle stop."
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

    def right_range_cb(self, msg: Range):
        raw = float(msg.range)
        self.right_range_raw = raw
        self.right_range = self.low_pass_filter(self.right_range, raw, self.left_range_filter_alpha)

    def front_range_cb(self, msg: Range):
        raw = float(msg.range)
        self.front_range_raw = raw
        self.front_range = self.low_pass_filter(self.front_range, raw, self.range_filter_alpha)

    def gyro_angle_cb(self, msg: Float32):
        self.current_yaw_deg = self.normalize_angle_deg(float(msg.data))
        self.gyro_ready = True

    def build_white_mask(self, bgr_img):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_low, self.s_low, self.v_low], dtype=np.uint8)
        upper = np.array([self.h_high, self.s_high, self.v_high], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return mask

    def get_red_search_roi(self, img_shape):
        h, w = img_shape[:2]

        y0 = int(h * self.red_roi_top_trim_ratio)
        x_trim = int(w * self.red_roi_side_trim_ratio)

        y0 = max(0, min(h - 1, y0))
        x_trim = max(0, min(max(0, (w // 2) - 2), x_trim))

        x0 = x_trim
        x1 = w - x_trim
        y1 = h

        if x1 <= x0:
            x0 = 0
            x1 = w

        return x0, y0, x1, y1

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

        x0, y0, x1, y1 = self.get_red_search_roi(bgr_img.shape)
        self.last_red_roi = (x0, y0, x1, y1)

        cropped_mask = np.zeros_like(mask)
        cropped_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        return cropped_mask

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

    def compute_box_approach_speed(self, distance: float) -> float:
        if not self.valid_range(distance):
            return self.box_approach_speed

        if distance <= self.box_stop_distance_m:
            return 0.0

        if distance >= self.box_slow_distance_m:
            return self.box_approach_max_speed

        ratio = (
            (distance - self.box_stop_distance_m) /
            (self.box_slow_distance_m - self.box_stop_distance_m)
        )
        speed = ratio * self.box_approach_max_speed

        if speed < 0.02:
            speed = 0.02

        return speed

    def start_measurement(self):
        if self.measurement_started:
            return
        self.measurement_started = True
        self.finish_wall_counter = 0
        self.front_obstacle_counter = 0
        self.left_side_detector.reset()
        self.right_side_detector.reset()
        self.left_side_info = self.left_side_detector.last_info
        self.right_side_info = self.right_side_detector.last_info
        self.get_logger().info("Started box detection during white-line following using side-sensor variation pattern.")

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
        self.get_logger().info("Initial box-search ignore finished. Normal side variation box detection enabled.")
        return False

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

    def update_side_variation_detectors(self, now_sec: float):
        startup_ignored = self.startup_ignore_active_now()
        left_enabled = self.measurement_started and (not startup_ignored) and (not self.same_side_is_ignored('LEFT'))
        right_enabled = self.measurement_started and (not startup_ignored) and (not self.same_side_is_ignored('RIGHT'))

        left_event, self.left_side_info = self.left_side_detector.update(
            self.left_range, now_sec, enabled=left_enabled and self.state == self.STATE_FOLLOW_LINE
        )
        right_event, self.right_side_info = self.right_side_detector.update(
            self.right_range, now_sec, enabled=right_enabled and self.state == self.STATE_FOLLOW_LINE
        )
        return left_event, right_event

    def choose_box_side(self, now_sec: float):
        left_event, right_event = self.update_side_variation_detectors(now_sec)

        if not self.measurement_started or self.state != self.STATE_FOLLOW_LINE:
            return None

        if self.startup_ignore_active_now():
            return None

        if not left_event and not right_event:
            return None

        if left_event and right_event:
            left_score = self.left_side_info['peak_excursion'] + 0.5 * self.left_side_info['excursion_area']
            right_score = self.right_side_info['peak_excursion'] + 0.5 * self.right_side_info['excursion_area']
            side = 'LEFT' if left_score >= right_score else 'RIGHT'
        elif left_event:
            side = 'LEFT'
        else:
            side = 'RIGHT'

        info = self.left_side_info if side == 'LEFT' else self.right_side_info
        self.get_logger().info(
            f"Variation-based box detection on {side}: "
            f"peak_drop={info['peak_excursion']:.3f}m area={info['excursion_area']:.3f} "
            f"baseline={info['baseline']:.3f} fast={info['fast']:.3f} "
            f"duration={info['duration']:.2f}s"
        )
        return side

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
        self.left_side_detector.reset()
        self.right_side_detector.reset()
        self.left_side_info = self.left_side_detector.last_info
        self.right_side_info = self.right_side_detector.last_info
        self.active_box_side = side
        self.box_stop_counter = 0
        self.box_drive_start_time = None
        self.red_lost_counter = 0
        self.reverse_start_time = None
        self.box_forward_before_turn_start_time = None
        self.box_stop_reached_time = None
        self.pick_retry_due_time = None
        self.box_turn_completed_time = None
        self.box_red_confirmed = False
        self.red_confirm_counter = 0

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

    def decrement_active_box_count(self):
        if self.active_box_side == 'LEFT':
            self.left_box_count = max(0, self.left_box_count - 1)
        elif self.active_box_side == 'RIGHT':
            self.right_box_count = max(0, self.right_box_count - 1)

    def abort_false_box_and_resume(self, reason: str):
        side = self.active_box_side
        if side is None:
            self.state = self.STATE_FOLLOW_LINE
            return

        self.decrement_active_box_count()
        self.box_stop_counter = 0
        self.box_drive_start_time = None
        self.red_lost_counter = 0
        self.reverse_start_time = None
        self.box_stop_reached_time = None
        self.pick_retry_due_time = None
        self.box_turn_completed_time = None
        self.box_red_confirmed = False
        self.red_confirm_counter = 0

        self.get_logger().warn(
            f"Aborting {side} box detour: {reason}. Corrected false {side} box count and returning to line."
        )

        self.start_relative_turn(
            self.opposite_side_sign(side) * self.turn_angle_90_deg,
            self.STATE_BOX_TURN_TO_RESUME,
            self.STATE_FOLLOW_LINE
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
                    self.box_turn_completed_time = self.get_clock().now()
                    self.red_lost_counter = 0
                    self.box_red_confirmed = False
                    self.red_confirm_counter = 0
                    self.get_logger().info(
                        f"Turned toward {self.active_box_side} box. Waiting for centered red-box confirmation."
                    )
                elif self.state == self.STATE_FOLLOW_LINE:
                    handled_side = self.active_box_side
                    self.box_stop_counter = 0
                    self.red_lost_counter = 0
                    self.box_drive_start_time = None
                    self.reverse_start_time = None
                    self.box_stop_reached_time = None
                    self.pick_retry_due_time = None
                    self.box_turn_completed_time = None
                    self.box_red_confirmed = False
                    self.red_confirm_counter = 0
                    self.active_box_side = None
                    if handled_side is not None:
                        self.start_same_box_ignore(handled_side)
                    self.left_side_detector.reset()
                    self.right_side_detector.reset()
                    self.left_side_info = self.left_side_detector.last_info
                    self.right_side_info = self.right_side_detector.last_info
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
                    self.box_turn_completed_time = self.get_clock().now()
                    self.red_lost_counter = 0
                    self.box_red_confirmed = False
                    self.red_confirm_counter = 0
                elif self.state == self.STATE_FOLLOW_LINE:
                    handled_side = self.active_box_side
                    self.box_stop_counter = 0
                    self.red_lost_counter = 0
                    self.box_drive_start_time = None
                    self.reverse_start_time = None
                    self.box_stop_reached_time = None
                    self.pick_retry_due_time = None
                    self.box_turn_completed_time = None
                    self.box_red_confirmed = False
                    self.red_confirm_counter = 0
                    self.active_box_side = None
                    if handled_side is not None:
                        self.start_same_box_ignore(handled_side)
                    self.left_side_detector.reset()
                    self.right_side_detector.reset()
                    self.left_side_info = self.left_side_detector.last_info
                    self.right_side_info = self.right_side_detector.last_info
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

        self.get_logger().info(
            f"Sending PickBox goal for side={self.active_box_side}, attempt={self.pick_retry_count + 1}"
        )

        future = self.pick_client.send_goal_async(
            goal_msg,
            feedback_callback=self.pick_feedback_callback
        )
        future.add_done_callback(self.pick_goal_response_callback)

    def pick_feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.pick_feedback_text = f"{fb.current_step} ({fb.progress:.2f})"

    def pick_goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.pick_in_progress = False
            self.pick_result_ready = True
            self.pick_result_success = False
            self.pick_result_message = f"Failed to send goal: {exc}"
            self.get_logger().error(self.pick_result_message)
            return

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

    def show_debug_windows_safe(self, vis, mask, bottom_mask, red_mask):
        if not self.show_debug_windows or self.imshow_failed:
            return
        try:
            cv2.imshow('task2_main_debug', vis)
            # cv2.imshow('task2_white_mask', mask)
            # cv2.imshow('task2_bottom_white_mask', bottom_mask)
            # cv2.imshow('task2_red_mask', red_mask)
            cv2.waitKey(1)
        except Exception as exc:
            self.imshow_failed = True
            self.get_logger().warn(f'OpenCV imshow disabled: {exc}')

    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]

        y0 = int(h * self.roi_y_start)
        roi = frame[y0:h, 0:w]
        mask = self.build_white_mask(roi)

        M = cv2.moments(mask)
        area = M["m00"]

        bh = int(h * self.bottom_strip_height_ratio)
        by0 = max(0, h - bh)
        bottom_roi = frame[by0:h, 0:w]
        bottom_mask = self.build_white_mask(bottom_roi)
        Mb = cv2.moments(bottom_mask)
        bottom_area = Mb["m00"]

        red_found, red_cx, red_cy, red_area, red_bbox, red_mask = self.detect_red_box(frame)

        twist = Twist()
        now_sec = self.get_clock().now().nanoseconds / 1e9

        if self.state in {
            self.STATE_LINE_CROSS_APPROACH,
            self.STATE_LINE_CROSS_WAIT_DISAPPEAR,
            self.STATE_LINE_CROSS_EXTRA_FORWARD,
            self.STATE_LINE_CROSS_TURN,
            self.STATE_LINE_CROSS_POST_WAIT,
        }:
            self.run_line_cross_sequence(bottom_area, twist)

        elif self.state == self.STATE_FOLLOW_LINE:

            # Exit white-line follower when front wall is detected below 0.40 m
            if self.valid_range(self.front_range_raw) and self.front_range_raw < self.front_wall_detect_distance:
                self.get_logger().info(
                    f"Front wall detected at {self.front_range_raw:.3f} m. "
                    f"Exiting white-line following and moving straight until "
                    f"{self.front_wall_stop_distance:.3f} m."
                )
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.state = self.STATE_FRONT_APPROACH

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

                side = self.choose_box_side(now_sec)
                if side is not None:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.start_box_detour(side)

        elif self.state == self.STATE_FRONT_APPROACH:
            # Move straight forward until front distance reaches 0.20 m, then stop
            if self.valid_range(self.front_range_raw):
                if self.front_range_raw > self.front_wall_stop_distance:
                    twist.linear.x = self.front_wall_approach_speed
                    twist.angular.z = 0.0
                else:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.state = self.STATE_TASK2_DONE
                    self.task2_done = True
                    self.stop_robot()
                    self.publish_task_done()
                    self.get_logger().info(
                        f"Front approach complete. Reached stop distance: "
                        f"{self.front_range_raw:.3f} m <= {self.front_wall_stop_distance:.3f} m"
                    )
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.get_logger().warn(
                    "Front range invalid during FRONT_APPROACH. Holding position."
                )

        elif self.state == self.STATE_BOX_FORWARD_BEFORE_TURN:
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

            if not self.box_red_confirmed:
                twist.linear.x = 0.0
                twist.angular.z = 0.0

                if red_found:
                    self.red_confirm_counter += 1
                else:
                    self.red_confirm_counter = 0

                if self.red_confirm_counter >= self.red_confirm_frames:
                    self.box_red_confirmed = True
                    self.red_lost_counter = 0
                    self.get_logger().info(
                        f"Red box confirmed in front for {self.active_box_side} side. Starting approach."
                    )
                elif self.box_turn_completed_time is not None:
                    confirm_elapsed = (self.get_clock().now() - self.box_turn_completed_time).nanoseconds / 1e9
                    if confirm_elapsed >= self.red_confirm_timeout_sec:
                        self.abort_false_box_and_resume(
                            f"no centered red box seen within {self.red_confirm_timeout_sec:.2f}s after turn"
                        )

            if self.state == self.STATE_BOX_DRIVE_TO_BOX and self.box_red_confirmed:
                speed_by_front = self.compute_box_approach_speed(front_dist)
                stop_now = speed_by_front == 0.0

                if red_found:
                    self.red_lost_counter = 0
                    error = float(red_cx - (w // 2))
                    ang = -self.red_kp * error
                    ang = self.clamp(ang, -self.red_max_angular, self.red_max_angular)

                    twist.linear.x = speed_by_front
                    twist.angular.z = ang if speed_by_front > 0.0 else 0.0
                else:
                    self.red_lost_counter += 1
                    twist.linear.x = 0.0
                    twist.angular.z = self.side_sign(self.active_box_side) * self.red_search_angular

                    if self.red_lost_counter > self.red_lost_frames_limit:
                        self.get_logger().info("Confirmed red box lost. Rotating slowly to reacquire target.")

                if stop_now:
                    self.box_stop_counter += 1
                else:
                    self.box_stop_counter = 0

                if self.box_stop_counter >= self.box_stop_frames:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.state = self.STATE_BOX_STOP_SETTLE
                    self.box_stop_reached_time = self.get_clock().now()
                    self.get_logger().info(
                        f"Reached {self.active_box_side} box using slow-stop logic. "
                        f"front={self.fmt_range(front_dist)} side={self.fmt_range(side_dist)}. "
                        f"Settling for {self.box_stop_settle_sec:.2f}s before pick."
                    )
                elif self.box_drive_start_time is not None:
                    elapsed = (self.get_clock().now() - self.box_drive_start_time).nanoseconds / 1e9
                    if elapsed >= self.box_drive_timeout_sec:
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                        self.state = self.STATE_BOX_STOP_SETTLE
                        self.box_stop_reached_time = self.get_clock().now()
                        self.get_logger().warn(
                            f"Box drive timeout. Proceeding to settle then pick. "
                            f"front={self.fmt_range(front_dist)} side={self.fmt_range(side_dist)} "
                            f"active_box={self.active_box_side}"
                        )

        elif self.state == self.STATE_BOX_STOP_SETTLE:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

            if self.box_stop_reached_time is not None:
                elapsed = (self.get_clock().now() - self.box_stop_reached_time).nanoseconds / 1e9
                if elapsed >= self.box_stop_settle_sec:
                    self.state = self.STATE_BOX_REQUEST_PICK
                    self.get_logger().info("Box settle complete. Sending pick request.")

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
                self.pick_retry_due_time = self.get_clock().now() + Duration(seconds=self.pick_retry_wait_sec)
                self.state = self.STATE_BOX_PICK_RETRY_WAIT
                self.get_logger().warn(
                    f"Retrying pick action after {self.pick_retry_wait_sec:.1f}s: "
                    f"attempt {self.pick_retry_count}/{self.pick_retry_limit}"
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

        elif self.state == self.STATE_BOX_PICK_RETRY_WAIT:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

            if self.pick_retry_due_time is not None and self.get_clock().now() >= self.pick_retry_due_time:
                self.pick_retry_due_time = None
                self.box_stop_reached_time = self.get_clock().now()
                self.state = self.STATE_BOX_STOP_SETTLE
                self.get_logger().info("Retry wait complete. Settling again before resending pick goal.")

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
                f"Ldrop={self.left_side_info.get('excursion', 0.0):.3f}/{self.left_side_info.get('peak_excursion', 0.0):.3f} "
                f"Rdrop={self.right_side_info.get('excursion', 0.0):.3f}/{self.right_side_info.get('peak_excursion', 0.0):.3f} "
                f"front_stop_counter={self.front_obstacle_counter}/{self.front_obstacle_stop_frames} "
                f"red_found={red_found} red_area={int(red_area)} "
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

        if self.last_red_roi is not None:
            rx0, ry0, rx1, ry1 = self.last_red_roi
            cv2.rectangle(vis, (rx0, ry0), (rx1 - 1, ry1 - 1), (0, 0, 180), 2)
            cv2.putText(vis, "RED SEARCH ROI", (rx0 + 8, max(25, ry0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 2)

        if red_found and red_bbox is not None:
            x, y, bw, bh = red_bbox
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cv2.circle(vis, (red_cx, red_cy), 6, (0, 0, 255), -1)
            cv2.line(vis, (w // 2, 0), (w // 2, h - 1), (0, 0, 255), 1)
            cv2.putText(vis, f"red_area={int(red_area)}", (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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

        cv2.putText(
            vis,
            f"Ldrop={self.left_side_info.get('excursion', 0.0):.3f} peak={self.left_side_info.get('peak_excursion', 0.0):.3f} "
            f"Rdrop={self.right_side_info.get('excursion', 0.0):.3f} peak={self.right_side_info.get('peak_excursion', 0.0):.3f}",
            (10, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 210, 120),
            2
        )

        cv2.putText(
            vis,
            f"var_th: start={self.side_variation_start_drop_m:.3f} peak={self.side_variation_peak_drop_m:.3f} end={self.side_variation_end_drop_m:.3f}",
            (10, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 210, 120),
            2
        )

        cv2.putText(
            vis,
            f"front_stop={self.front_obstacle_counter}/{self.front_obstacle_stop_frames}",
            (10, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 220, 150),
            2
        )

        cv2.putText(
            vis,
            f"front_exit={self.front_wall_detect_distance:.2f} front_final={self.front_wall_stop_distance:.2f}",
            (10, 275),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 220, 150),
            2
        )

        cv2.putText(
            vis,
            f"LEFT count={self.left_box_count} RIGHT count={self.right_box_count}",
            (10, 305),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2
        )

        cv2.putText(
            vis,
            f"boxes_completed={self.boxes_completed}/{self.target_box_count}",
            (10, 335),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 220, 120),
            2
        )

        cv2.putText(
            vis,
            f"active_box={self.active_box_side} red_found={red_found} red_ok={self.box_red_confirmed}",
            (10, 365),
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
            (10, 395),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 255, 255),
            2
        )

        cv2.putText(
            vis,
            f"Ltrack={self.left_side_info.get('tracking', False)} Rtrack={self.right_side_info.get('tracking', False)} red_confirm={self.red_confirm_counter}/{self.red_confirm_frames}",
            (10, 425),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2
        )

        cv2.putText(
            vis,
            f"cmd v={twist.linear.x:.2f} w={twist.angular.z:.2f}",
            (10, 455),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        self.show_debug_windows_safe(vis, mask, bottom_mask, red_mask)

        self.frame_count += 1
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds > 1_000_000_000:
            self.last_log_time = now
            self.get_logger().info(
                f"fps~{self.frame_count} state={self.state} "
                f"yaw={yaw_txt} target={tgt_txt} "
                f"main_area={int(area)} bottom_area={int(bottom_area)} red_area={int(red_area)} "
                f"left={self.display_range(self.left_range)} "
                f"right={self.display_range(self.right_range)} "
                f"front={self.display_range(self.front_range)} "
                f"startup_ignore={startup_ignore_txt} "
                f"ignore_side={ignore_txt} "
                f"Ldrop/peak=({self.left_side_info.get('excursion', 0.0):.3f},{self.left_side_info.get('peak_excursion', 0.0):.3f}) "
                f"Rdrop/peak=({self.right_side_info.get('excursion', 0.0):.3f},{self.right_side_info.get('peak_excursion', 0.0):.3f}) "
                f"counts(L,R)=({self.left_box_count},{self.right_box_count}) "
                f"boxes_completed={self.boxes_completed}/{self.target_box_count} "
                f"active_box={self.active_box_side} red_ok={self.box_red_confirmed} "
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

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()