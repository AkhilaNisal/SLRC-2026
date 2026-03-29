#!/usr/bin/env python3
"""
Task 1 — Camera-only maze navigation.

Navigates the maze using *only* the camera and wheel encoders
(no ToF sensors, no gyro / IMU).  Floor-colour segmentation determines
corridor centering, front-wall detection, and side-opening detection.
Encoder differential drives 90° turns and maze heading tracking.
"""

import json
import math
from enum import Enum

import cv2
import gpiod
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, Float32, Int32, Int64, String


# ── state enums ──────────────────────────────────────────────────

class NavState(Enum):
    FOLLOW = 0
    CREEP_TO_JUNCTION = 1
    STOP_AND_DECIDE = 2
    TURNING = 3
    TAG_BLINK = 4
    POST_TURN = 5


class TurnDir(Enum):
    LEFT = 'L'
    RIGHT = 'R'
    BACK = 'B'
    NONE = 'N'


# ── node ─────────────────────────────────────────────────────────

class Task1CameraNode(Node):

    def __init__(self):
        super().__init__('task1')
        self._declare_params()
        self._read_params()
        self._init_state()
        self._init_ros()
        self._init_led()
        self.get_logger().info('Task1 camera maze node started')

    # ── parameter declaration ────────────────────────────────────

    def _declare_params(self):
        # topics
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('image_topic', '/camera/image/image_color')
        self.declare_parameter('gyro_angle_topic', '/gyro_angle')
        self.declare_parameter('left_steps_topic', '/stepper/left_steps_total')
        self.declare_parameter('right_steps_topic', '/stepper/right_steps_total')
        self.declare_parameter('distance_since_reset_topic', '/distance_since_reset')
        self.declare_parameter('reset_distance_topic', '/reset_distance')
        self.declare_parameter('junction_count_topic', '/junction_count')
        self.declare_parameter('dead_end_count_topic', '/dead_end_count')
        self.declare_parameter('cell_count_topic', '/cell_count')
        self.declare_parameter('apriltag_topic', '/apriltag/decoded')

        # rates
        self.declare_parameter('control_rate_hz', 20.0)

        # ── floor segmentation (HSV) ────────────────────────────
        self.declare_parameter('floor_h_low', 0)
        self.declare_parameter('floor_s_low', 0)
        self.declare_parameter('floor_v_low', 180)
        self.declare_parameter('floor_h_high', 180)
        self.declare_parameter('floor_s_high', 70)
        self.declare_parameter('floor_v_high', 255)
        self.declare_parameter('use_clahe', True)

        # ── camera ROI fractions (of frame height / width) ──────
        # centering ROI (bottom strip)
        self.declare_parameter('center_roi_y_start', 0.70)
        self.declare_parameter('center_roi_y_end', 1.00)
        # forward-look ROI (middle strip — blocked detection)
        self.declare_parameter('forward_roi_y_start', 0.25)
        self.declare_parameter('forward_roi_y_end', 0.55)
        # side-opening ROIs (left/right columns in a mid strip)
        self.declare_parameter('side_roi_y_start', 0.35)
        self.declare_parameter('side_roi_y_end', 0.65)
        self.declare_parameter('side_roi_width_ratio', 0.30)

        # ── vision thresholds ───────────────────────────────────
        # floor-area ratio in forward ROI below which front is "blocked"
        self.declare_parameter('front_blocked_ratio', 0.15)
        # floor-area ratio in side ROI above which side is "open"
        self.declare_parameter('side_open_ratio', 0.25)
        # minimum floor pixels for valid centering
        self.declare_parameter('center_min_area', 800)
        # confirm frames before declaring blocked / open
        self.declare_parameter('front_blocked_confirm', 4)
        self.declare_parameter('side_open_confirm', 3)
        self.declare_parameter('front_clear_confirm', 2)

        # ── corridor centering ──────────────────────────────────
        self.declare_parameter('center_kp', 0.006)
        self.declare_parameter('center_max_angular', 1.0)

        # ── speeds ──────────────────────────────────────────────
        self.declare_parameter('max_forward_speed', 0.12)
        self.declare_parameter('slow_forward_speed', 0.06)
        self.declare_parameter('min_forward_speed', 0.03)
        self.declare_parameter('search_linear', 0.04)
        self.declare_parameter('search_angular', 0.30)

        # ── encoder geometry ─────────────────────────────────
        self.declare_parameter('encoder_wheel_radius', 0.0325)
        self.declare_parameter('encoder_wheel_base', 0.20)
        self.declare_parameter('encoder_steps_per_rev', 200)
        self.declare_parameter('encoder_microsteps', 16)

        # ── camera-to-axle offset ────────────────────────────────
        # distance the robot must creep forward after a visual
        # detection so the wheel axle lines up with the junction
        self.declare_parameter('camera_axle_offset_m', 0.45)
        self.declare_parameter('creep_speed', 0.05)

        # ── turn controller ─────────────────────────────────────
        self.declare_parameter('turn_angular_speed', 0.45)
        self.declare_parameter('turn_slow_angular_speed', 0.22)
        self.declare_parameter('turn_slowdown_error_deg', 18.0)
        self.declare_parameter('turn_tolerance_deg', 4.0)
        self.declare_parameter('turn_settle_cycles', 4)
        self.declare_parameter('post_turn_forward_sec', 0.50)
        self.declare_parameter('junction_cooldown_sec', 0.80)
        self.declare_parameter('prefer_left_first', True)
        # encoder-based turn angle limit
        self.declare_parameter('turn_max_encoder_deg', 110.0)

        # ── output shaping ──────────────────────────────────────
        self.declare_parameter('max_angular', 0.8)
        self.declare_parameter('angular_slew_per_cycle', 0.12)

        # ── junction / cell / maze ──────────────────────────────
        self.declare_parameter('junction_min_distance_m', 0.12)
        self.declare_parameter('cell_length_m', 0.40)
        self.declare_parameter('maze_rows', 3)
        self.declare_parameter('maze_cols', 6)
        self.declare_parameter('start_row', 0)
        self.declare_parameter('start_col', 0)
        self.declare_parameter('start_facing', 'E')

        # ── AprilTag + LED ──────────────────────────────────────
        self.declare_parameter('led_chip_name', 'gpiochip4')
        self.declare_parameter('led_pin', 26)
        self.declare_parameter('blink_half_period_sec', 0.3)

        # ── debug ───────────────────────────────────────────────
        self.declare_parameter('debug_logs', True)
        self.declare_parameter('debug_every_n', 10)

    # ── parameter read ───────────────────────────────────────────

    def _read_params(self):
        g = self.get_parameter

        # topics
        self.cmd_vel_topic = g('cmd_vel_topic').value
        self.image_topic = g('image_topic').value
        self.gyro_angle_topic = g('gyro_angle_topic').value
        self.left_steps_topic = g('left_steps_topic').value
        self.right_steps_topic = g('right_steps_topic').value
        self.distance_since_reset_topic = g('distance_since_reset_topic').value
        self.reset_distance_topic = g('reset_distance_topic').value
        self.junction_count_topic = g('junction_count_topic').value
        self.dead_end_count_topic = g('dead_end_count_topic').value
        self.cell_count_topic = g('cell_count_topic').value
        self.apriltag_topic = g('apriltag_topic').value

        self.control_rate_hz = float(g('control_rate_hz').value)

        # floor segmentation
        self.floor_h_low = int(g('floor_h_low').value)
        self.floor_s_low = int(g('floor_s_low').value)
        self.floor_v_low = int(g('floor_v_low').value)
        self.floor_h_high = int(g('floor_h_high').value)
        self.floor_s_high = int(g('floor_s_high').value)
        self.floor_v_high = int(g('floor_v_high').value)
        self.use_clahe = bool(g('use_clahe').value)

        # ROIs
        self.center_roi_y_start = float(g('center_roi_y_start').value)
        self.center_roi_y_end = float(g('center_roi_y_end').value)
        self.forward_roi_y_start = float(g('forward_roi_y_start').value)
        self.forward_roi_y_end = float(g('forward_roi_y_end').value)
        self.side_roi_y_start = float(g('side_roi_y_start').value)
        self.side_roi_y_end = float(g('side_roi_y_end').value)
        self.side_roi_width_ratio = float(g('side_roi_width_ratio').value)

        # vision thresholds
        self.front_blocked_ratio = float(g('front_blocked_ratio').value)
        self.side_open_ratio = float(g('side_open_ratio').value)
        self.center_min_area = int(g('center_min_area').value)
        self.front_blocked_confirm = int(g('front_blocked_confirm').value)
        self.side_open_confirm = int(g('side_open_confirm').value)
        self.front_clear_confirm = int(g('front_clear_confirm').value)

        # centering
        self.center_kp = float(g('center_kp').value)
        self.center_max_angular = float(g('center_max_angular').value)

        # speeds
        self.max_forward_speed = float(g('max_forward_speed').value)
        self.slow_forward_speed = float(g('slow_forward_speed').value)
        self.min_forward_speed = float(g('min_forward_speed').value)
        self.search_linear = float(g('search_linear').value)
        self.search_angular = float(g('search_angular').value)

        # encoder
        enc_r = float(g('encoder_wheel_radius').value)
        enc_b = float(g('encoder_wheel_base').value)
        enc_spr = int(g('encoder_steps_per_rev').value)
        enc_us = int(g('encoder_microsteps').value)
        self.enc_wheel_base = enc_b
        self.enc_meters_per_step = (2.0 * math.pi * enc_r) / (enc_spr * enc_us)

        # camera-axle offset
        self.camera_axle_offset_m = float(g('camera_axle_offset_m').value)
        self.creep_speed = float(g('creep_speed').value)

        # turn
        self.turn_angular_speed = float(g('turn_angular_speed').value)
        self.turn_slow_angular_speed = float(g('turn_slow_angular_speed').value)
        self.turn_slowdown_error_deg = float(g('turn_slowdown_error_deg').value)
        self.turn_tolerance_deg = float(g('turn_tolerance_deg').value)
        self.turn_settle_cycles = int(g('turn_settle_cycles').value)
        self.post_turn_forward_sec = float(g('post_turn_forward_sec').value)
        self.junction_cooldown_sec = float(g('junction_cooldown_sec').value)
        self.prefer_left_first = bool(g('prefer_left_first').value)
        self.turn_max_encoder_deg = float(g('turn_max_encoder_deg').value)

        # output
        self.max_angular = float(g('max_angular').value)
        self.angular_slew_per_cycle = float(g('angular_slew_per_cycle').value)

        # junction / cell / maze
        self.junction_min_distance_m = float(g('junction_min_distance_m').value)
        self.cell_length_m = float(g('cell_length_m').value)
        self.maze_rows = int(g('maze_rows').value)
        self.maze_cols = int(g('maze_cols').value)
        self.start_row = int(g('start_row').value)
        self.start_col = int(g('start_col').value)
        self.start_facing = str(g('start_facing').value)

        # LED
        self.led_chip_name = str(g('led_chip_name').value)
        self.led_pin = int(g('led_pin').value)
        self.blink_half_period_sec = float(g('blink_half_period_sec').value)

        # debug
        self.debug_logs = bool(g('debug_logs').value)
        self.debug_every_n = int(g('debug_every_n').value)

    # ── state initialisation ─────────────────────────────────────

    def _init_state(self):
        self.bridge = CvBridge()

        # vision state
        self.latest_frame = None
        self.floor_cx = None            # centroid x of floor in centering ROI
        self.front_floor_ratio = 1.0    # floor area ratio in forward ROI
        self.left_floor_ratio = 0.0     # floor area ratio in left side ROI
        self.right_floor_ratio = 0.0    # floor area ratio in right side ROI

        # confirmation counters
        self.front_blocked_count = 0
        self.front_clear_count = 0
        self.left_open_count = 0
        self.right_open_count = 0
        self.visually_front_blocked = False
        self.visually_left_open = False
        self.visually_right_open = False

        self.prev_angular_cmd = 0.0

        # encoder heading
        self.left_steps = None
        self.right_steps = None
        self.prev_left_steps = None
        self.prev_right_steps = None
        self.encoder_yaw_rad = 0.0

        # nav state
        self.nav_state = NavState.FOLLOW
        self.turn_direction = TurnDir.NONE
        self.turn_settle_counter = 0
        self.post_turn_cycles = 0
        self.post_turn_total_cycles = 0
        self.last_junction_time = None

        # creep-to-junction state (camera-axle offset)
        self.creep_start_distance = 0.0
        # snapshot of side-open flags at creep entry
        self.creep_left_open = False
        self.creep_right_open = False

        # gyro turn tracking (used only during turns)
        self.current_gyro_deg = None
        self.turn_gyro_start_deg = None

        # encoder-based turn tracking
        self.turn_start_left_steps = None
        self.turn_start_right_steps = None

        # junction / cell counting
        self.distance_since_reset_m = 0.0
        self.junction_count = 0
        self.dead_end_count = 0
        self.cell_count = 0
        self.segment_cells_counted = 0

        # maze memory
        self.robot_row = self.start_row
        self.robot_col = self.start_col
        self.visited = {(self.robot_row, self.robot_col)}
        _facing_offsets = {'E': 0.0, 'N': -90.0, 'W': -180.0, 'S': 90.0}
        self.maze_heading_offset = _facing_offsets.get(self.start_facing, 0.0)

        # AprilTag
        self.seen_tag_ids: set = set()
        self._tag_blink_queued = False
        self._led_line = None
        self._blink_count = 0
        self._blink_timer = None

        # debug
        self._dbg_counter = 0

    # ── ROS wiring ───────────────────────────────────────────────

    def _init_ros(self):
        qos = qos_profile_sensor_data

        # subscribers
        self.create_subscription(Image, self.image_topic, self._image_cb, qos)
        self.create_subscription(Float32, self.gyro_angle_topic, self._gyro_cb, qos)
        self.create_subscription(Int64, self.left_steps_topic, self._left_steps_cb, 10)
        self.create_subscription(Int64, self.right_steps_topic, self._right_steps_cb, 10)
        self.create_subscription(Float32, self.distance_since_reset_topic,
                                 self._distance_cb, 10)
        self.create_subscription(String, self.apriltag_topic, self._apriltag_cb, 10)

        # publishers
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.reset_distance_pub = self.create_publisher(Empty, self.reset_distance_topic, 10)
        self.junction_count_pub = self.create_publisher(Int32, self.junction_count_topic, 10)
        self.dead_end_count_pub = self.create_publisher(Int32, self.dead_end_count_topic, 10)
        self.cell_count_pub = self.create_publisher(Int32, self.cell_count_topic, 10)

        # control timer
        self.timer = self.create_timer(1.0 / self.control_rate_hz, self._control_loop)

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _clamp(x, lo, hi):
        return max(lo, min(hi, x))

    @staticmethod
    def _wrap_deg(a):
        while a > 180.0:
            a -= 360.0
        while a < -180.0:
            a += 360.0
        return a

    def _angle_err(self, target, current):
        return self._wrap_deg(target - current)

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _slew(self, target, prev, step):
        if target > prev + step:
            return prev + step
        if target < prev - step:
            return prev - step
        return target

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _recent_junction(self):
        if self.last_junction_time is None:
            return False
        return (self._now_sec() - self.last_junction_time) < self.junction_cooldown_sec

    def _fmt(self, x):
        return 'None' if x is None else f'{x:.3f}'

    def _dbg(self, msg):
        if self.debug_logs:
            self.get_logger().info(msg)

    def _dbg_periodic(self, msg):
        if self.debug_logs and (self._dbg_counter % self.debug_every_n) == 0:
            self.get_logger().info(msg)

    # ── vision: floor segmentation ───────────────────────────────

    def _floor_mask(self, bgr):
        """Return binary mask of floor pixels using HSV thresholding."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

        lower = np.array([self.floor_h_low, self.floor_s_low, self.floor_v_low],
                         dtype=np.uint8)
        upper = np.array([self.floor_h_high, self.floor_s_high, self.floor_v_high],
                         dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def _process_frame(self, bgr):
        """Extract navigation signals from the camera frame."""
        h, w = bgr.shape[:2]
        mask = self._floor_mask(bgr)

        # ── centering ROI (bottom strip) ─────────────────────────
        cy0 = int(h * self.center_roi_y_start)
        cy1 = int(h * self.center_roi_y_end)
        center_roi = mask[cy0:cy1, :]
        M = cv2.moments(center_roi)
        area = M['m00']
        if area >= self.center_min_area:
            self.floor_cx = int(M['m10'] / area)
        else:
            self.floor_cx = None

        # ── forward ROI (middle strip — blocked detection) ───────
        fy0 = int(h * self.forward_roi_y_start)
        fy1 = int(h * self.forward_roi_y_end)
        forward_roi = mask[fy0:fy1, :]
        fwd_total = forward_roi.shape[0] * forward_roi.shape[1]
        fwd_floor = cv2.countNonZero(forward_roi)
        self.front_floor_ratio = fwd_floor / max(1, fwd_total)

        # ── side ROIs (left / right columns in mid strip) ────────
        sy0 = int(h * self.side_roi_y_start)
        sy1 = int(h * self.side_roi_y_end)
        sw = int(w * self.side_roi_width_ratio)

        left_roi = mask[sy0:sy1, 0:sw]
        right_roi = mask[sy0:sy1, w - sw:w]

        l_total = left_roi.shape[0] * left_roi.shape[1]
        r_total = right_roi.shape[0] * right_roi.shape[1]
        self.left_floor_ratio = cv2.countNonZero(left_roi) / max(1, l_total)
        self.right_floor_ratio = cv2.countNonZero(right_roi) / max(1, r_total)

        # ── confirmation logic ───────────────────────────────────
        # front blocked
        if self.front_floor_ratio < self.front_blocked_ratio:
            self.front_blocked_count += 1
            self.front_clear_count = 0
        else:
            self.front_clear_count += 1
            self.front_blocked_count = 0

        if self.front_blocked_count >= self.front_blocked_confirm:
            self.visually_front_blocked = True
        if self.front_clear_count >= self.front_clear_confirm:
            self.visually_front_blocked = False

        # left open
        if self.left_floor_ratio >= self.side_open_ratio:
            self.left_open_count += 1
        else:
            self.left_open_count = 0
        self.visually_left_open = self.left_open_count >= self.side_open_confirm

        # right open
        if self.right_floor_ratio >= self.side_open_ratio:
            self.right_open_count += 1
        else:
            self.right_open_count = 0
        self.visually_right_open = self.right_open_count >= self.side_open_confirm

        return w

    # ── sensor callbacks ─────────────────────────────────────────

    def _image_cb(self, msg: Image):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'Image conversion failed: {e}')

    def _gyro_cb(self, msg: Float32):
        self.current_gyro_deg = self._wrap_deg(float(msg.data))

    def _left_steps_cb(self, msg: Int64):
        self.left_steps = int(msg.data)
        self._update_encoder_heading()

    def _right_steps_cb(self, msg: Int64):
        self.right_steps = int(msg.data)
        self._update_encoder_heading()

    def _distance_cb(self, msg: Float32):
        self.distance_since_reset_m = float(msg.data)

    def _apriltag_cb(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        incoming = set(payload.get('tag_ids', []))
        new_ids = incoming - self.seen_tag_ids
        if not new_ids:
            return
        self.seen_tag_ids |= new_ids
        decoded = payload.get('decoded_by_order', {})
        self.get_logger().info(
            f'[APRILTAG] New: {sorted(new_ids)} | '
            f'total={len(self.seen_tag_ids)}/8 | decoded={decoded}')
        if self.nav_state == NavState.FOLLOW:
            self._start_blink()
        else:
            self._tag_blink_queued = True

    # ── encoder heading tracking ──────────────────────────────

    def _update_encoder_heading(self):
        if self.left_steps is None or self.right_steps is None:
            return
        if self.prev_left_steps is None:
            self.prev_left_steps = self.left_steps
            self.prev_right_steps = self.right_steps
            return
        dl = (self.left_steps - self.prev_left_steps) * self.enc_meters_per_step
        dr = (self.right_steps - self.prev_right_steps) * self.enc_meters_per_step
        self.prev_left_steps = self.left_steps
        self.prev_right_steps = self.right_steps
        dtheta = (dr - dl) / self.enc_wheel_base
        self.encoder_yaw_rad += dtheta
        while self.encoder_yaw_rad > math.pi:
            self.encoder_yaw_rad -= 2.0 * math.pi
        while self.encoder_yaw_rad < -math.pi:
            self.encoder_yaw_rad += 2.0 * math.pi

    # ── corridor centering steering ──────────────────────────────

    def _center_term(self, frame_w):
        """Proportional steering from floor centroid offset."""
        if self.floor_cx is None:
            return None  # signal: line lost
        error = float(self.floor_cx - frame_w // 2)
        ang = -self.center_kp * error
        return self._clamp(ang, -self.center_max_angular, self.center_max_angular)

    # ── forward speed ────────────────────────────────────────────

    def _forward_speed(self):
        """Adaptive speed: slow when front is getting crowded."""
        if self.visually_front_blocked:
            return 0.0
        # ramp between max and slow based on forward floor ratio
        r = self.front_floor_ratio
        # ratio 0.15..0.40  →  maps to slow..max
        lo, hi = self.front_blocked_ratio, self.front_blocked_ratio + 0.25
        t = self._clamp((r - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        return self.slow_forward_speed + t * (self.max_forward_speed - self.slow_forward_speed)

    # ── camera-to-axle offset (creep) ────────────────────────────

    def _begin_creep(self):
        """Start creeping forward so the wheel axle reaches the junction."""
        self.creep_start_distance = self.distance_since_reset_m
        # snapshot current side-open states so we remember what was seen
        self.creep_left_open = self.visually_left_open
        self.creep_right_open = self.visually_right_open
        self.nav_state = NavState.CREEP_TO_JUNCTION
        self._dbg(
            f'[CREEP_START] dist={self.distance_since_reset_m:.3f} '
            f'offset={self.camera_axle_offset_m:.3f} '
            f'L_open={self.creep_left_open} R_open={self.creep_right_open}')

    def _execute_creep(self):
        """Drive slowly forward until the axle offset is covered."""
        driven = self.distance_since_reset_m - self.creep_start_distance

        # continuously latch any side openings seen during the creep
        if self.visually_left_open:
            self.creep_left_open = True
        if self.visually_right_open:
            self.creep_right_open = True

        if driven >= self.camera_axle_offset_m:
            self._stop()
            self._dbg(
                f'[CREEP_DONE] driven={driven:.3f} '
                f'L={self.creep_left_open} R={self.creep_right_open} -> STOP_AND_DECIDE')
            self.nav_state = NavState.STOP_AND_DECIDE
            return
        t = Twist()
        t.linear.x = self.creep_speed
        self.cmd_pub.publish(t)

    # ── turning ──────────────────────────────────────────────────

    def _begin_turn(self, turn_dir: TurnDir):
        if turn_dir == TurnDir.LEFT:
            delta = 90.0
        elif turn_dir == TurnDir.RIGHT:
            delta = -90.0
        elif turn_dir == TurnDir.BACK:
            delta = 180.0
        else:
            return False

        self.turn_direction = turn_dir
        self.turn_settle_counter = 0
        self.nav_state = NavState.TURNING
        self.prev_angular_cmd = 0.0

        # record start state for both gyro and encoder
        self.turn_gyro_start_deg = self.current_gyro_deg
        self.turn_start_left_steps = self.left_steps
        self.turn_start_right_steps = self.right_steps

        if self.turn_gyro_start_deg is not None:
            self.get_logger().info(
                f'Turn {turn_dir.value} | delta={delta:.0f}° (gyro primary)')
        else:
            self.get_logger().info(
                f'Turn {turn_dir.value} | delta={delta:.0f}° (encoder fallback)')
        return True

    def _encoder_turn_deg(self):
        """Estimate how many degrees the robot has turned using encoder steps."""
        if (self.turn_start_left_steps is None or self.left_steps is None
                or self.turn_start_right_steps is None or self.right_steps is None):
            return 0.0
        dl = (self.left_steps - self.turn_start_left_steps) * self.enc_meters_per_step
        dr = (self.right_steps - self.turn_start_right_steps) * self.enc_meters_per_step
        return math.degrees((dr - dl) / self.enc_wheel_base)

    def _finish_turn(self, source: str):
        enc_deg = self._encoder_turn_deg()
        self.get_logger().info(
            f'Turn done ({source}) enc={enc_deg:.1f}° -> POST_TURN')
        self._stop()
        self.nav_state = NavState.POST_TURN
        self.turn_direction = TurnDir.NONE
        self.prev_angular_cmd = 0.0
        self.last_junction_time = self._now_sec()
        self.post_turn_cycles = 0
        self.post_turn_total_cycles = max(
            1, int(self.post_turn_forward_sec * self.control_rate_hz))

    def _execute_turn(self):
        # determine desired turn sign (+1 = left / CCW, -1 = right / CW)
        if self.turn_direction == TurnDir.LEFT:
            sign = 1.0
        elif self.turn_direction == TurnDir.RIGHT:
            sign = -1.0
        elif self.turn_direction == TurnDir.BACK:
            sign = 1.0  # U-turn goes left
        else:
            self._stop()
            return

        target_deg = 90.0 if self.turn_direction != TurnDir.BACK else 180.0

        # ── encoder hard-limit safety ─────────────────────────
        enc_deg = self._encoder_turn_deg()
        if abs(enc_deg) >= self.turn_max_encoder_deg:
            self._finish_turn('encoder limit')
            return

        # ── gyro-based turn (primary) ─────────────────────────
        if self.current_gyro_deg is not None and self.turn_gyro_start_deg is not None:
            gyro_delta = abs(self._wrap_deg(self.current_gyro_deg - self.turn_gyro_start_deg))
            remaining = target_deg - gyro_delta
            if remaining <= self.turn_tolerance_deg:
                self._finish_turn('gyro')
                return
            spd = self.turn_angular_speed
            if remaining < self.turn_slowdown_error_deg:
                spd = self.turn_slow_angular_speed
            t = Twist()
            t.angular.z = sign * spd
            self.cmd_pub.publish(t)
            return

        # ── encoder fallback (no gyro) ────────────────────────
        remaining = target_deg - abs(enc_deg)
        if remaining <= self.turn_tolerance_deg:
            self._finish_turn('encoder')
            return
        spd = self.turn_angular_speed
        if remaining < self.turn_slowdown_error_deg:
            spd = self.turn_slow_angular_speed
        t = Twist()
        t.angular.z = sign * spd
        self.cmd_pub.publish(t)

    # ── decision logic ───────────────────────────────────────────

    def _choose_turn(self):
        # use latched side-open flags captured during creep
        left_open = self.creep_left_open or self.visually_left_open
        right_open = self.creep_right_open or self.visually_right_open

        self._count_block_event(left_open, right_open)

        self._dbg(
            f'[DECIDE] fwd_ratio={self.front_floor_ratio:.3f} '
            f'L_ratio={self.left_floor_ratio:.3f} open={left_open} '
            f'R_ratio={self.right_floor_ratio:.3f} open={right_open}')

        if left_open and right_open:
            lc = self._neighbor_cell(TurnDir.LEFT)
            rc = self._neighbor_cell(TurnDir.RIGHT)
            lv = self._visited_or_oob(lc)
            rv = self._visited_or_oob(rc)
            if lv and not rv:
                choice = TurnDir.RIGHT
            elif rv and not lv:
                choice = TurnDir.LEFT
            else:
                choice = TurnDir.LEFT if self.prefer_left_first else TurnDir.RIGHT
            self.get_logger().info(
                f'Both open | L_vis={lv} R_vis={rv} -> {choice.value}')
            return choice

        if left_open:
            self.get_logger().info('Left open -> LEFT')
            return TurnDir.LEFT
        if right_open:
            self.get_logger().info('Right open -> RIGHT')
            return TurnDir.RIGHT

        self.get_logger().info('Dead end -> BACK')
        return TurnDir.BACK

    # ── junction / cell / maze ───────────────────────────────────

    def _reset_distance(self):
        self.reset_distance_pub.publish(Empty())
        self.distance_since_reset_m = 0.0
        self.segment_cells_counted = 0

    def _can_count(self):
        return self.distance_since_reset_m >= self.junction_min_distance_m

    def _publish_counts(self):
        j, d = Int32(), Int32()
        j.data, d.data = self.junction_count, self.dead_end_count
        self.junction_count_pub.publish(j)
        self.dead_end_count_pub.publish(d)

    def _publish_cell(self):
        c = Int32()
        c.data = self.cell_count
        self.cell_count_pub.publish(c)

    def _count_pass_junction(self, lo, ro):
        if not self._can_count():
            return
        self.junction_count += 1
        self._publish_counts()
        self.get_logger().info(
            f'[JUNC_PASS] #{self.junction_count} dist={self.distance_since_reset_m:.3f} '
            f'L={lo} R={ro}')
        self._reset_distance()

    def _count_block_event(self, lo, ro):
        if not self._can_count():
            return
        if lo or ro:
            self.junction_count += 1
            self._publish_counts()
            self.get_logger().info(
                f'[JUNC_BLOCK] #{self.junction_count} L={lo} R={ro}')
        else:
            self.dead_end_count += 1
            self._publish_counts()
            self.get_logger().info(f'[DEAD_END] #{self.dead_end_count}')
        self._reset_distance()

    def _update_cells(self):
        if self.cell_length_m <= 0.0:
            return
        cells_now = int(self.distance_since_reset_m / self.cell_length_m)
        if cells_now > self.segment_cells_counted:
            new = cells_now - self.segment_cells_counted
            self.segment_cells_counted = cells_now
            self.cell_count += new
            self._publish_cell()
            for _ in range(new):
                self._advance_maze()
            self.get_logger().info(
                f'[CELL] total={self.cell_count} seg={self.segment_cells_counted} '
                f'dist={self.distance_since_reset_m:.3f}')

    # ── maze memory ──────────────────────────────────────────────

    def _snap_cardinal(self):
        rel = self._wrap_deg(
            math.degrees(self.encoder_yaw_rad) + self.maze_heading_offset)
        return self._wrap_deg(round(rel / 90.0) * 90.0)

    @staticmethod
    def _cardinal_delta(c):
        if c is None:
            return (0, 0)
        c = round(c)
        if c == 0:   return (0, 1)
        if c == 90:  return (-1, 0)
        if c == -90: return (1, 0)
        return (0, -1)

    def _advance_maze(self):
        card = self._snap_cardinal()
        dr, dc = self._cardinal_delta(card)
        nr, nc = self.robot_row + dr, self.robot_col + dc
        if 0 <= nr < self.maze_rows and 0 <= nc < self.maze_cols:
            self.robot_row, self.robot_col = nr, nc
        self.visited.add((self.robot_row, self.robot_col))
        self.get_logger().info(
            f'[MAZE] ({self.robot_row},{self.robot_col}) h={card} '
            f'vis={len(self.visited)}/{self.maze_rows*self.maze_cols}')

    def _neighbor_cell(self, td):
        card = self._snap_cardinal()
        if card is None:
            return None
        if td == TurnDir.LEFT:
            nh = self._wrap_deg(card + 90.0)
        elif td == TurnDir.RIGHT:
            nh = self._wrap_deg(card - 90.0)
        elif td == TurnDir.BACK:
            nh = self._wrap_deg(card + 180.0)
        else:
            return None
        dr, dc = self._cardinal_delta(nh)
        return (self.robot_row + dr, self.robot_col + dc)

    def _visited_or_oob(self, cell):
        if cell is None:
            return True
        r, c = cell
        if not (0 <= r < self.maze_rows and 0 <= c < self.maze_cols):
            return True
        return (r, c) in self.visited

    # ── LED blink ────────────────────────────────────────────────

    def _init_led(self):
        try:
            chip = gpiod.Chip(self.led_chip_name)
            self._led_line = chip.get_line(self.led_pin)
            self._led_line.request(
                consumer='task1_led',
                type=gpiod.LINE_REQ_DIR_OUT,
                default_vals=[0])
            self.get_logger().info(
                f'LED on {self.led_chip_name} pin {self.led_pin}')
        except Exception as exc:
            self._led_line = None
            self.get_logger().warn(f'LED init failed: {exc}')

    def _set_led(self, on):
        if self._led_line is None:
            return
        try:
            self._led_line.set_value(1 if on else 0)
        except Exception:
            pass

    def _start_blink(self):
        if self._blink_timer is not None:
            self._blink_timer.cancel()
        self._blink_count = 0
        self.nav_state = NavState.TAG_BLINK
        self._set_led(True)
        self._blink_timer = self.create_timer(
            self.blink_half_period_sec, self._blink_step)
        self.get_logger().info('AprilTag -> blinking LED ×2')

    def _blink_step(self):
        self._blink_count += 1
        self._set_led(self._blink_count % 2 == 0)
        if self._blink_count >= 3:
            self._set_led(False)
            self._blink_timer.cancel()
            self._blink_timer = None
            self.nav_state = NavState.FOLLOW
            self.get_logger().info('Blink done -> FOLLOW')

    # ── follow motion ────────────────────────────────────────────

    def _follow(self, frame_w):
        """Corridor centering (camera only).  Returns True if stopped."""
        twist = Twist()

        center = self._center_term(frame_w)

        if center is None:
            # floor lost — slow search spin
            twist.linear.x = self.search_linear
            twist.angular.z = self.search_angular
            self.cmd_pub.publish(twist)
            return False

        raw_ang = center
        raw_ang = self._clamp(raw_ang, -self.max_angular, self.max_angular)
        ang = self._slew(raw_ang, self.prev_angular_cmd, self.angular_slew_per_cycle)
        ang = self._clamp(ang, -self.max_angular, self.max_angular)
        self.prev_angular_cmd = ang

        speed = self._forward_speed()
        if speed <= 0.0:
            self._stop()
            return True

        # reduce speed when steering hard
        steer_scale = 1.0 - 0.30 * min(1.0, abs(ang) / max(1e-6, self.max_angular))
        speed = max(self.min_forward_speed, speed * steer_scale)

        twist.linear.x = speed
        twist.angular.z = ang
        self.cmd_pub.publish(twist)
        return False

    # ── control loop ─────────────────────────────────────────────

    def _control_loop(self):
        self._dbg_counter += 1
        self._update_cells()

        # ── process latest camera frame ──────────────────────────
        frame_w = 640  # fallback
        if self.latest_frame is not None:
            frame_w = self._process_frame(self.latest_frame)

        # ── state machine ────────────────────────────────────────
        if self.nav_state == NavState.TURNING:
            self._execute_turn()
            return

        if self.nav_state == NavState.TAG_BLINK:
            self._stop()
            return

        if self.nav_state == NavState.CREEP_TO_JUNCTION:
            self._execute_creep()
            return

        if self.nav_state == NavState.POST_TURN:
            # brief forward after turn to clear junction visually
            self.post_turn_cycles += 1
            if self.post_turn_cycles >= self.post_turn_total_cycles:
                self.nav_state = NavState.FOLLOW
                self.get_logger().info('POST_TURN done -> FOLLOW')
                # reset visual confirmation so we don't immediately re-trigger
                self.front_blocked_count = 0
                self.front_clear_count = self.front_clear_confirm
                self.visually_front_blocked = False
                return
            t = Twist()
            t.linear.x = self.slow_forward_speed
            self.cmd_pub.publish(t)
            return

        if self.nav_state == NavState.STOP_AND_DECIDE:
            self._stop()
            choice = self._choose_turn()
            self._begin_turn(choice)
            return

        # ── FOLLOW ───────────────────────────────────────────────
        if self._tag_blink_queued:
            self._tag_blink_queued = False
            self._start_blink()
            return

        if self.latest_frame is None:
            # no camera data yet
            self._stop()
            return

        # count pass-through junctions (side openings while front is clear)
        if (not self.visually_front_blocked
                and (self.visually_left_open or self.visually_right_open)
                and not self._recent_junction()):
            self._count_pass_junction(self.visually_left_open,
                                      self.visually_right_open)
            self.last_junction_time = self._now_sec()

        self._dbg_periodic(
            f'[FOLLOW] fwd={self.front_floor_ratio:.2f} blocked={self.visually_front_blocked} '
            f'L={self.left_floor_ratio:.2f} open={self.visually_left_open} '
            f'R={self.right_floor_ratio:.2f} open={self.visually_right_open} '
            f'cx={self._fmt(self.floor_cx)} dist={self.distance_since_reset_m:.3f} '
            f'cells={self.cell_count}')

        stopped = self._follow(frame_w)
        if stopped and not self._recent_junction():
            self._begin_creep()
            return

    # ── shutdown ─────────────────────────────────────────────────

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


# ── entry point ──────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = Task1CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop_robot()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
