#!/usr/bin/env python3

import json
import math
from enum import Enum

import gpiod
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Range, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Int32, Int64, Empty, String


# ── state enums ──────────────────────────────────────────────────

class NavState(Enum):
    FOLLOW = 0
    REFRESH_FOR_DECISION = 1
    STOP_AND_DECIDE = 2
    TURNING = 3
    TAG_BLINK = 4


class TurnDir(Enum):
    LEFT = 'L'
    RIGHT = 'R'
    BACK = 'B'
    NONE = 'N'


# ── node ─────────────────────────────────────────────────────────

class Task1MazeNode(Node):

    # ── construction ─────────────────────────────────────────────

    def __init__(self):
        super().__init__('task1')
        self._declare_params()
        self._read_params()
        self._init_state()
        self._init_ros()
        self._init_led()
        self.get_logger().info('Task1 maze node started')

    # ── parameter declaration ────────────────────────────────────

    def _declare_params(self):
        # topics
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('left_range_topic', '/robocop/ds_left')
        self.declare_parameter('front_range_topic', '/robocop/ds_front')
        self.declare_parameter('right_range_topic', '/robocop/ds_right')
        self.declare_parameter('gyro_angle_topic', '/gyro_angle')
        self.declare_parameter('distance_since_reset_topic', '/distance_since_reset')
        self.declare_parameter('reset_distance_topic', '/reset_distance')
        self.declare_parameter('junction_count_topic', '/junction_count')
        self.declare_parameter('dead_end_count_topic', '/dead_end_count')
        self.declare_parameter('cell_count_topic', '/cell_count')
        self.declare_parameter('left_steps_topic', '/stepper/left_steps_total')
        self.declare_parameter('right_steps_topic', '/stepper/right_steps_total')
        self.declare_parameter('imu_topic', '/imu/data_raw')
        self.declare_parameter('apriltag_topic', '/apriltag/decoded')

        # rates
        self.declare_parameter('control_rate_hz', 20.0)

        # speeds
        self.declare_parameter('max_forward_speed', 0.14)
        self.declare_parameter('transition_forward_speed', 0.08)
        self.declare_parameter('blind_forward_speed', 0.06)
        self.declare_parameter('min_forward_speed', 0.03)

        # front stop / slowdown
        self.declare_parameter('front_stop_distance', 0.25)
        self.declare_parameter('front_slow_distance', 0.35)
        self.declare_parameter('front_hard_slow_distance', 0.30)
        self.declare_parameter('front_stop_hysteresis', 0.01)

        # wall handling
        self.declare_parameter('wall_detect_distance', 0.30)
        self.declare_parameter('wall_clear_distance', 0.35)
        self.declare_parameter('turn_open_distance', 0.34)
        self.declare_parameter('min_valid_range', 0.04)
        self.declare_parameter('max_valid_range', 16.0)
        self.declare_parameter('wall_loss_confirm_cycles', 5)

        # wall PD gains
        self.declare_parameter('center_kp', 3.5)
        self.declare_parameter('center_kd', 2.2)

        # heading-hold gains
        self.declare_parameter('heading_kp', 0.030)
        self.declare_parameter('heading_kd', 0.010)
        self.declare_parameter('heading_weight_both', 0.20)
        self.declare_parameter('heading_weight_missing', 1.00)
        self.declare_parameter('heading_capture_alpha', 0.12)
        self.declare_parameter('heading_error_limit_deg', 25.0)
        self.declare_parameter('gyro_valid_timeout_sec', 0.50)

        # encoder heading
        self.declare_parameter('encoder_wheel_radius', 0.0325)
        self.declare_parameter('encoder_wheel_base', 0.20)
        self.declare_parameter('encoder_steps_per_rev', 200)
        self.declare_parameter('encoder_microsteps', 16)
        self.declare_parameter('imu_fusion_alpha', 0.05)

        # fusion shaping
        self.declare_parameter('wall_weight_both', 1.00)
        self.declare_parameter('wall_weight_near_front', 0.35)
        self.declare_parameter('near_front_fusion_distance', 0.30)

        # output shaping
        self.declare_parameter('max_angular', 0.8)
        self.declare_parameter('angular_deadband', 0.03)
        self.declare_parameter('angular_slew_per_cycle', 0.10)

        # filtering / edge handling
        self.declare_parameter('side_filter_alpha', 0.22)
        self.declare_parameter('max_side_jump', 0.14)
        self.declare_parameter('corridor_width_alpha', 0.10)

        # decision refresh
        self.declare_parameter('decision_required_samples', 5)
        self.declare_parameter('decision_filter_alpha', 0.35)

        # feed-forward wall escape
        self.declare_parameter('ff_enabled', True)
        self.declare_parameter('ff_trigger_delta', 0.005)
        self.declare_parameter('ff_trigger_cycles', 2)
        self.declare_parameter('ff_turn_mag', 0.35)
        self.declare_parameter('ff_hold_cycles', 6)
        self.declare_parameter('ff_cooldown_cycles', 5)
        self.declare_parameter('ff_front_min_distance', 0.1)
        self.declare_parameter('ff_edge_lock_block', True)
        self.declare_parameter('ff_heading_suppress_gain', 0.4)

        # junction counting
        self.declare_parameter('junction_min_distance_m', 0.12)
        self.declare_parameter('count_pass_junctions', True)
        self.declare_parameter('count_blocked_junctions', True)
        self.declare_parameter('count_dead_ends', True)

        # cell counting
        self.declare_parameter('cell_length_m', 0.40)

        # maze memory
        self.declare_parameter('maze_rows', 3)
        self.declare_parameter('maze_cols', 6)
        self.declare_parameter('start_row', 0)
        self.declare_parameter('start_col', 0)
        self.declare_parameter('start_facing', 'E')

        # debug
        self.declare_parameter('debug_logs', True)
        self.declare_parameter('debug_every_n_cycles', 5)

        # LED
        self.declare_parameter('led_chip_name', 'gpiochip4')
        self.declare_parameter('led_pin', 26)
        self.declare_parameter('blink_half_period_sec', 0.3)

        # turn controller
        self.declare_parameter('turn_angular_speed', 0.45)
        self.declare_parameter('turn_slow_angular_speed', 0.22)
        self.declare_parameter('turn_slowdown_error_deg', 18.0)
        self.declare_parameter('turn_tolerance_deg', 3.0)
        self.declare_parameter('turn_settle_cycles', 4)
        self.declare_parameter('post_turn_forward_time_sec', 0.35)
        self.declare_parameter('junction_cooldown_sec', 0.80)
        self.declare_parameter('prefer_left_first', True)

    # ── parameter read ───────────────────────────────────────────

    def _read_params(self):
        p = self.get_parameter

        # topics
        self.cmd_vel_topic = p('cmd_vel_topic').value
        self.left_range_topic = p('left_range_topic').value
        self.front_range_topic = p('front_range_topic').value
        self.right_range_topic = p('right_range_topic').value
        self.gyro_angle_topic = p('gyro_angle_topic').value
        self.distance_since_reset_topic = p('distance_since_reset_topic').value
        self.reset_distance_topic = p('reset_distance_topic').value
        self.junction_count_topic = p('junction_count_topic').value
        self.dead_end_count_topic = p('dead_end_count_topic').value
        self.cell_count_topic = p('cell_count_topic').value
        self.left_steps_topic = p('left_steps_topic').value
        self.right_steps_topic = p('right_steps_topic').value
        self.imu_topic = p('imu_topic').value
        self.apriltag_topic = p('apriltag_topic').value

        # rates
        self.control_rate_hz = float(p('control_rate_hz').value)

        # speeds
        self.max_forward_speed = float(p('max_forward_speed').value)
        self.transition_forward_speed = float(p('transition_forward_speed').value)
        self.blind_forward_speed = float(p('blind_forward_speed').value)
        self.min_forward_speed = float(p('min_forward_speed').value)

        # front stop / slowdown
        self.front_stop_distance = float(p('front_stop_distance').value)
        self.front_slow_distance = float(p('front_slow_distance').value)
        self.front_hard_slow_distance = float(p('front_hard_slow_distance').value)
        self.front_stop_hysteresis = float(p('front_stop_hysteresis').value)

        # wall handling
        self.wall_detect_distance = float(p('wall_detect_distance').value)
        self.wall_clear_distance = float(p('wall_clear_distance').value)
        self.turn_open_distance = float(p('turn_open_distance').value)
        self.min_valid_range = float(p('min_valid_range').value)
        self.max_valid_range = float(p('max_valid_range').value)
        self.wall_loss_confirm_cycles = int(p('wall_loss_confirm_cycles').value)

        # wall PD gains
        self.center_kp = float(p('center_kp').value)
        self.center_kd = float(p('center_kd').value)

        # heading-hold
        self.heading_kp = float(p('heading_kp').value)
        self.heading_kd = float(p('heading_kd').value)
        self.heading_weight_both = float(p('heading_weight_both').value)
        self.heading_weight_missing = float(p('heading_weight_missing').value)
        self.heading_capture_alpha = float(p('heading_capture_alpha').value)
        self.heading_error_limit_deg = float(p('heading_error_limit_deg').value)
        self.gyro_valid_timeout_sec = float(p('gyro_valid_timeout_sec').value)

        # encoder heading
        enc_radius = float(p('encoder_wheel_radius').value)
        self.enc_wheel_base = float(p('encoder_wheel_base').value)
        enc_spr = int(p('encoder_steps_per_rev').value)
        enc_us = int(p('encoder_microsteps').value)
        self.imu_fusion_alpha = float(p('imu_fusion_alpha').value)
        self.enc_meters_per_step = (2.0 * math.pi * enc_radius) / (enc_spr * enc_us)

        # fusion shaping
        self.wall_weight_both = float(p('wall_weight_both').value)
        self.wall_weight_near_front = float(p('wall_weight_near_front').value)
        self.near_front_fusion_distance = float(p('near_front_fusion_distance').value)

        # output shaping
        self.max_angular = float(p('max_angular').value)
        self.angular_deadband = float(p('angular_deadband').value)
        self.angular_slew_per_cycle = float(p('angular_slew_per_cycle').value)

        # filtering
        self.side_filter_alpha = float(p('side_filter_alpha').value)
        self.max_side_jump = float(p('max_side_jump').value)
        self.corridor_width_alpha = float(p('corridor_width_alpha').value)

        # decision refresh
        self.decision_required_samples = int(p('decision_required_samples').value)
        self.decision_filter_alpha = float(p('decision_filter_alpha').value)

        # feed-forward
        self.ff_enabled = bool(p('ff_enabled').value)
        self.ff_trigger_delta = float(p('ff_trigger_delta').value)
        self.ff_trigger_cycles = int(p('ff_trigger_cycles').value)
        self.ff_turn_mag = float(p('ff_turn_mag').value)
        self.ff_hold_cycles = int(p('ff_hold_cycles').value)
        self.ff_cooldown_cycles = int(p('ff_cooldown_cycles').value)
        self.ff_front_min_distance = float(p('ff_front_min_distance').value)
        self.ff_edge_lock_block = bool(p('ff_edge_lock_block').value)
        self.ff_heading_suppress_gain = float(p('ff_heading_suppress_gain').value)

        # junction counting
        self.junction_min_distance_m = float(p('junction_min_distance_m').value)
        self.count_pass_junctions = bool(p('count_pass_junctions').value)
        self.count_blocked_junctions = bool(p('count_blocked_junctions').value)
        self.count_dead_ends = bool(p('count_dead_ends').value)

        # cell counting
        self.cell_length_m = float(p('cell_length_m').value)

        # maze memory
        self.maze_rows = int(p('maze_rows').value)
        self.maze_cols = int(p('maze_cols').value)
        self.start_row = int(p('start_row').value)
        self.start_col = int(p('start_col').value)
        self.start_facing = str(p('start_facing').value)

        # debug
        self.debug_logs = bool(p('debug_logs').value)
        self.debug_every_n_cycles = int(p('debug_every_n_cycles').value)

        # LED
        self.led_chip_name = str(p('led_chip_name').value)
        self.led_pin = int(p('led_pin').value)
        self.blink_half_period_sec = float(p('blink_half_period_sec').value)

        # turn controller
        self.turn_angular_speed = float(p('turn_angular_speed').value)
        self.turn_slow_angular_speed = float(p('turn_slow_angular_speed').value)
        self.turn_slowdown_error_deg = float(p('turn_slowdown_error_deg').value)
        self.turn_tolerance_deg = float(p('turn_tolerance_deg').value)
        self.turn_settle_cycles = int(p('turn_settle_cycles').value)
        self.post_turn_forward_time_sec = float(p('post_turn_forward_time_sec').value)
        self.junction_cooldown_sec = float(p('junction_cooldown_sec').value)
        self.prefer_left_first = bool(p('prefer_left_first').value)

    # ── state init ───────────────────────────────────────────────

    def _init_state(self):
        # sensor readings
        self.left_range = None
        self.front_range = None
        self.right_range = None
        self.left_valid = False
        self.front_valid = False
        self.right_valid = False

        # wall presence with hysteresis
        self.left_wall_present = False
        self.right_wall_present = False
        self.left_loss_counter = 0
        self.right_loss_counter = 0

        # edge lock (freeze side value on large jumps)
        self.left_edge_locked = False
        self.right_edge_locked = False

        # PD memory
        self.prev_center_error = 0.0
        self.prev_heading_error = 0.0
        self.prev_angular_cmd = 0.0
        self.last_stable_angular = 0.0

        # corridor width estimator
        self.corridor_width_est = None

        # follow mode label
        self.mode = 'INIT'
        self.stopped = False

        # gyro / yaw
        self.current_yaw_deg = None
        self.last_yaw_deg = None
        self.target_yaw_deg = None
        self.last_gyro_msg_time = None

        # encoder heading
        self.left_steps = None
        self.right_steps = None
        self.prev_left_steps = None
        self.prev_right_steps = None
        self.encoder_yaw_rad = 0.0
        self.imu_yaw_rad = None
        self.fused_yaw_rad = None
        self.fused_yaw_initialized = False

        # high-level nav
        self.nav_state = NavState.FOLLOW
        self.turn_direction = TurnDir.NONE
        self.turn_target_yaw_deg = None
        self.turn_settle_counter = 0
        self.post_turn_forward_cycles = 0
        self.last_junction_time = None
        self._back_turn_leg2_pending = False

        # decision refresh accumulators
        self.left_decision_range = None
        self.right_decision_range = None
        self.left_decision_samples = 0
        self.right_decision_samples = 0

        # feed-forward tracking
        self.prev_left_ff_range = None
        self.prev_right_ff_range = None
        self.left_toward_wall_count = 0
        self.right_toward_wall_count = 0
        self.ff_active_dir = TurnDir.NONE
        self.ff_hold_counter = 0
        self.ff_cooldown_counter = 0

        # debug helpers
        self.debug_cycle_counter = 0
        self.last_left_raw = None
        self.last_right_raw = None
        self.last_front_raw = None

        # junction counting
        self.distance_since_reset_m = 0.0
        self.junction_count = 0
        self.dead_end_count = 0
        self.pass_junction_count = 0
        self.blocked_junction_count = 0
        self.pending_block_event = False
        self.pending_block_event_distance = 0.0

        # cell counting
        self.cell_count = 0
        self.segment_cells_counted = 0

        # maze memory
        self.robot_row = self.start_row
        self.robot_col = self.start_col
        self.visited = {(self.robot_row, self.robot_col)}
        self.initial_gyro_deg = None
        facing_offsets = {'E': 0.0, 'N': -90.0, 'W': -180.0, 'S': 90.0}
        self.maze_heading_offset = facing_offsets.get(self.start_facing, 0.0)

        # apriltag / LED blink
        self.seen_tag_ids: set = set()
        self._tag_blink_queued = False
        self._led_line = None
        self._blink_count = 0
        self._blink_timer = None

    # ── ROS subscriptions / publishers / timer ───────────────────

    def _init_ros(self):
        qos = qos_profile_sensor_data

        # subscriptions
        self.create_subscription(Range, self.left_range_topic, self._left_cb, qos)
        self.create_subscription(Range, self.front_range_topic, self._front_cb, qos)
        self.create_subscription(Range, self.right_range_topic, self._right_cb, qos)
        self.create_subscription(Float32, self.gyro_angle_topic, self._gyro_cb, qos)
        self.create_subscription(Imu, self.imu_topic, self._imu_cb, qos)
        self.create_subscription(Int64, self.left_steps_topic, self._left_steps_cb, 10)
        self.create_subscription(Int64, self.right_steps_topic, self._right_steps_cb, 10)
        self.create_subscription(Float32, self.distance_since_reset_topic, self._distance_cb, 10)
        self.create_subscription(String, self.apriltag_topic, self._apriltag_cb, 10)

        # publishers
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.reset_distance_pub = self.create_publisher(Empty, self.reset_distance_topic, 10)
        self.junction_count_pub = self.create_publisher(Int32, self.junction_count_topic, 10)
        self.dead_end_count_pub = self.create_publisher(Int32, self.dead_end_count_topic, 10)
        self.cell_count_pub = self.create_publisher(Int32, self.cell_count_topic, 10)

        # control timer
        self.create_timer(1.0 / self.control_rate_hz, self._control_loop)

        self.get_logger().info(
            f'left={self.left_range_topic}  front={self.front_range_topic}  '
            f'right={self.right_range_topic}  gyro={self.gyro_angle_topic}'
        )
        self.get_logger().info(
            f'junction counting enabled | min_spacing={self.junction_min_distance_m:.3f} m'
        )
        self.get_logger().info(
            f'cell counting enabled | cell_length={self.cell_length_m:.3f} m'
        )

    # ── tiny utilities ───────────────────────────────────────────

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

    def _angle_err_deg(self, target, current):
        return self._wrap_deg(target - current)

    def _valid(self, x):
        return (x is not None
                and not math.isnan(x)
                and not math.isinf(x)
                and self.min_valid_range <= x <= self.max_valid_range)

    def _ema(self, old, new, alpha):
        return new if old is None else alpha * new + (1.0 - alpha) * old

    def _deadband(self, x, db):
        return 0.0 if abs(x) < db else x

    def _slew(self, target, prev, step):
        return self._clamp(target, prev - step, prev + step)

    def _gyro_fresh(self):
        if self.last_gyro_msg_time is None:
            return False
        dt = (self.get_clock().now() - self.last_gyro_msg_time).nanoseconds * 1e-9
        return dt <= self.gyro_valid_timeout_sec

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _front_blocked(self):
        return (self.front_valid
                and self.front_range is not None
                and self.front_range <= self.front_stop_distance + self.front_stop_hysteresis)

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
        if not self.debug_logs:
            return
        if self.debug_every_n_cycles <= 1 or (self.debug_cycle_counter % self.debug_every_n_cycles) == 0:
            self.get_logger().info(msg)

    # ── LED ──────────────────────────────────────────────────────

    def _init_led(self):
        try:
            chip = gpiod.Chip(self.led_chip_name)
            self._led_line = chip.get_line(self.led_pin)
            self._led_line.request(consumer='task1_led',
                                   type=gpiod.LINE_REQ_DIR_OUT,
                                   default_vals=[0])
            self.get_logger().info(
                f'Red LED on {self.led_chip_name} pin {self.led_pin}')
        except Exception as exc:
            self._led_line = None
            self.get_logger().warn(
                f'LED init failed ({self.led_chip_name} pin {self.led_pin}): {exc}. '
                'Tag detection will log only — no physical blink.')

    def _set_led(self, on: bool):
        if self._led_line is None:
            return
        try:
            self._led_line.set_value(1 if on else 0)
        except Exception as exc:
            self.get_logger().warn(f'LED set_value error: {exc}')

    # ── sensor callbacks ─────────────────────────────────────────

    def _left_cb(self, msg: Range):
        r = float(msg.range)
        self.last_left_raw = r

        if not self._valid(r):
            self._dbg(f'[LEFT_CB] invalid raw={self._fmt(r)}')
            return

        if self.nav_state == NavState.REFRESH_FOR_DECISION:
            self.left_decision_range = self._ema(
                self.left_decision_range, r, self.decision_filter_alpha)
            self.left_decision_samples += 1
            self.left_range = self.left_decision_range
            self.left_valid = True
            return

        if self.left_range is None:
            self.left_range = r
        elif self.left_edge_locked:
            pass
        else:
            if abs(r - self.left_range) > self.max_side_jump:
                self.left_edge_locked = True
            else:
                self.left_range = self._ema(self.left_range, r, self.side_filter_alpha)

        self.left_valid = True

    def _front_cb(self, msg: Range):
        r = float(msg.range)
        self.last_front_raw = r
        if self._valid(r):
            self.front_range = r
            self.front_valid = True

    def _right_cb(self, msg: Range):
        r = float(msg.range)
        self.last_right_raw = r

        if not self._valid(r):
            self._dbg(f'[RIGHT_CB] invalid raw={self._fmt(r)}')
            return

        if self.nav_state == NavState.REFRESH_FOR_DECISION:
            self.right_decision_range = self._ema(
                self.right_decision_range, r, self.decision_filter_alpha)
            self.right_decision_samples += 1
            self.right_range = self.right_decision_range
            self.right_valid = True
            return

        if self.right_range is None:
            self.right_range = r
        elif self.right_edge_locked:
            pass
        else:
            if abs(r - self.right_range) > self.max_side_jump:
                self.right_edge_locked = True
            else:
                self.right_range = self._ema(self.right_range, r, self.side_filter_alpha)

        self.right_valid = True

    def _gyro_cb(self, msg: Float32):
        yaw_deg = self._wrap_deg(float(msg.data))
        self.last_yaw_deg = self.current_yaw_deg

        if self.fused_yaw_rad is not None:
            self.current_yaw_deg = self._wrap_deg(math.degrees(self.fused_yaw_rad))
        else:
            self.current_yaw_deg = yaw_deg

        self.last_gyro_msg_time = self.get_clock().now()

        if self.target_yaw_deg is None:
            self.target_yaw_deg = self.current_yaw_deg

        if self.initial_gyro_deg is None:
            self.initial_gyro_deg = self.current_yaw_deg
            self.get_logger().info(
                f'[MAZE] Initial gyro captured: {self.current_yaw_deg:.1f} deg')

    def _imu_cb(self, msg: Imu):
        q = msg.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.imu_yaw_rad = math.atan2(siny, cosy)
        self._update_fused_heading()

    def _left_steps_cb(self, msg: Int64):
        self.left_steps = int(msg.data)
        self._update_encoder_heading()

    def _right_steps_cb(self, msg: Int64):
        self.right_steps = int(msg.data)
        self._update_encoder_heading()

    def _distance_cb(self, msg: Float32):
        self.distance_since_reset_m = float(msg.data)

    # ── encoder + IMU fusion ─────────────────────────────────────

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

        self.encoder_yaw_rad += (dr - dl) / self.enc_wheel_base
        self.encoder_yaw_rad = math.atan2(
            math.sin(self.encoder_yaw_rad), math.cos(self.encoder_yaw_rad))
        self._update_fused_heading()

    def _update_fused_heading(self):
        if self.imu_yaw_rad is None:
            self.fused_yaw_rad = self.encoder_yaw_rad
            return

        if not self.fused_yaw_initialized:
            self.encoder_yaw_rad = self.imu_yaw_rad
            self.fused_yaw_rad = self.imu_yaw_rad
            self.fused_yaw_initialized = True
            self.get_logger().info(
                f'[FUSION] Initialized at {math.degrees(self.imu_yaw_rad):.1f} deg')
            return

        err = math.atan2(
            math.sin(self.imu_yaw_rad - self.encoder_yaw_rad),
            math.cos(self.imu_yaw_rad - self.encoder_yaw_rad))
        self.fused_yaw_rad = self.encoder_yaw_rad + self.imu_fusion_alpha * err
        self.fused_yaw_rad = math.atan2(
            math.sin(self.fused_yaw_rad), math.cos(self.fused_yaw_rad))

    # ── wall presence with hysteresis ────────────────────────────

    def _update_wall_presence(self):
        for side in ('left', 'right'):
            valid = getattr(self, f'{side}_valid')
            rng = getattr(self, f'{side}_range')
            present = getattr(self, f'{side}_wall_present')
            loss_ctr = getattr(self, f'{side}_loss_counter')

            if not valid:
                continue

            if present:
                if rng > self.wall_clear_distance:
                    loss_ctr += 1
                    if loss_ctr >= self.wall_loss_confirm_cycles:
                        present = False
                        loss_ctr = 0
                else:
                    loss_ctr = 0
            else:
                if rng < self.wall_detect_distance:
                    present = True
                    loss_ctr = 0

            setattr(self, f'{side}_wall_present', present)
            setattr(self, f'{side}_loss_counter', loss_ctr)

    # ── forward speed computation ────────────────────────────────

    def _compute_forward_speed(self, mode):
        if not self.front_valid or self.front_range is None:
            if mode == 'GYRO_ONLY':
                return self.blind_forward_speed
            if mode == 'TRANSITION':
                return self.transition_forward_speed
            return min(0.08, self.max_forward_speed)

        if self.front_range <= self.front_stop_distance + self.front_stop_hysteresis:
            return 0.0

        if mode == 'GYRO_ONLY':
            cap = self.blind_forward_speed
        elif mode == 'TRANSITION':
            cap = self.transition_forward_speed
        else:
            cap = self.max_forward_speed

        if self.front_range <= self.front_hard_slow_distance:
            span = max(1e-6, self.front_hard_slow_distance - self.front_stop_distance)
            ratio = self._clamp(
                (self.front_range - self.front_stop_distance) / span, 0.0, 1.0)
            low = self.min_forward_speed
            high = min(cap, 0.07)
            return self._clamp(low + ratio * (high - low), low, cap)

        if self.front_range <= self.front_slow_distance:
            span = max(1e-6, self.front_slow_distance - self.front_hard_slow_distance)
            ratio = self._clamp(
                (self.front_range - self.front_hard_slow_distance) / span, 0.0, 1.0)
            low = min(cap, 0.07)
            return self._clamp(low + ratio * (cap - low), self.min_forward_speed, cap)

        return cap

    # ── heading-hold ─────────────────────────────────────────────

    def _update_target_heading(self, mode):
        if not self._gyro_fresh() or self.current_yaw_deg is None:
            return
        if mode == 'BOTH_WALLS':
            if self.target_yaw_deg is None:
                self.target_yaw_deg = self.current_yaw_deg
            else:
                err = self._angle_err_deg(self.current_yaw_deg, self.target_yaw_deg)
                self.target_yaw_deg = self._wrap_deg(
                    self.target_yaw_deg + self.heading_capture_alpha * err)

    def _compute_heading_term(self, mode):
        if (not self._gyro_fresh()
                or self.current_yaw_deg is None
                or self.target_yaw_deg is None):
            return 0.0

        err = self._clamp(
            self._angle_err_deg(self.target_yaw_deg, self.current_yaw_deg),
            -self.heading_error_limit_deg, self.heading_error_limit_deg)
        deriv = (err - self.prev_heading_error) * self.control_rate_hz
        self.prev_heading_error = err

        term = self.heading_kp * err + self.heading_kd * deriv
        weight = (self.heading_weight_both if mode == 'BOTH_WALLS'
                  else self.heading_weight_missing)
        return term * weight

    # ── feed-forward wall escape ─────────────────────────────────

    def _reset_ff(self):
        self.left_toward_wall_count = 0
        self.right_toward_wall_count = 0
        self.prev_left_ff_range = self.left_range
        self.prev_right_ff_range = self.right_range

    def _can_ff(self):
        return (self.ff_enabled
                and self.front_valid
                and self.front_range is not None
                and self.front_range > self.ff_front_min_distance
                and self.ff_cooldown_counter <= 0)

    def _start_ff(self, direction: TurnDir, reason: str):
        self.ff_active_dir = direction
        self.ff_hold_counter = self.ff_hold_cycles
        self.ff_cooldown_counter = self.ff_cooldown_cycles
        self._dbg(f'[FF_START] dir={direction.value} reason={reason}')

    def _compute_ff_term(self):
        # active pulse
        if self.ff_hold_counter > 0:
            self.ff_hold_counter -= 1
            if self.ff_active_dir == TurnDir.RIGHT:
                ang = -self.ff_turn_mag
            elif self.ff_active_dir == TurnDir.LEFT:
                ang = self.ff_turn_mag
            else:
                ang = 0.0
            if self.ff_hold_counter == 0:
                self._dbg(f'[FF_END] dir={self.ff_active_dir.value}')
                self.ff_active_dir = TurnDir.NONE
            return ang

        if self.ff_cooldown_counter > 0:
            self.ff_cooldown_counter -= 1

        left_ok = self.left_wall_present and self.left_valid and self.left_range is not None
        right_ok = self.right_wall_present and self.right_valid and self.right_range is not None
        if self.ff_edge_lock_block:
            if self.left_edge_locked:
                left_ok = False
            if self.right_edge_locked:
                right_ok = False

        if not self._can_ff():
            self._reset_ff()
            return 0.0

        if left_ok and not right_ok:
            prev = self.prev_left_ff_range
            if prev is not None and self.left_range < prev - self.ff_trigger_delta:
                self.left_toward_wall_count += 1
            else:
                self.left_toward_wall_count = max(0, self.left_toward_wall_count - 1)
            self.prev_left_ff_range = self.left_range
            self.prev_right_ff_range = self.right_range
            self.right_toward_wall_count = 0
            if self.left_toward_wall_count >= self.ff_trigger_cycles:
                self.left_toward_wall_count = 0
                self._start_ff(TurnDir.RIGHT, 'pushing_toward_left_wall')
                return -self.ff_turn_mag
            return 0.0

        if right_ok and not left_ok:
            prev = self.prev_right_ff_range
            if prev is not None and self.right_range < prev - self.ff_trigger_delta:
                self.right_toward_wall_count += 1
            else:
                self.right_toward_wall_count = max(0, self.right_toward_wall_count - 1)
            self.prev_right_ff_range = self.right_range
            self.prev_left_ff_range = self.left_range
            self.left_toward_wall_count = 0
            if self.right_toward_wall_count >= self.ff_trigger_cycles:
                self.right_toward_wall_count = 0
                self._start_ff(TurnDir.LEFT, 'pushing_toward_right_wall')
                return self.ff_turn_mag
            return 0.0

        self._reset_ff()
        return 0.0

    # ── wall steering ────────────────────────────────────────────

    def _compute_wall_term(self):
        left = self.left_wall_present and not self.left_edge_locked
        right = self.right_wall_present and not self.right_edge_locked

        if left and right and self.left_range is not None and self.right_range is not None:
            w = self.left_range + self.right_range
            if self.corridor_width_est is None:
                self.corridor_width_est = w
            else:
                a = self.corridor_width_alpha
                self.corridor_width_est = a * w + (1.0 - a) * self.corridor_width_est

        if left and right:
            self.mode = 'BOTH_WALLS'
            error = self.left_range - self.right_range
            deriv = (error - self.prev_center_error) * self.control_rate_hz
            self.prev_center_error = error
            ang = self.center_kp * error + self.center_kd * deriv
            self.last_stable_angular = 0.85 * self.last_stable_angular + 0.15 * ang
            return ang

        self.mode = 'GYRO_ONLY'
        self.prev_center_error = 0.0
        self.last_stable_angular *= 0.90
        return 0.0

    # ── turning ──────────────────────────────────────────────────

    def _begin_turn(self, turn_dir: TurnDir):
        if not self._gyro_fresh() or self.current_yaw_deg is None:
            self.get_logger().warn('Cannot begin turn: gyro not fresh')
            return False

        # For BACK (180°), split into two 90° legs to avoid ±180 wrap ambiguity
        if turn_dir == TurnDir.BACK:
            self._back_turn_leg2_pending = True
            delta = 90.0  # first leg: always turn left 90°
        else:
            self._back_turn_leg2_pending = False
            delta = {TurnDir.LEFT: 90.0, TurnDir.RIGHT: -90.0}.get(turn_dir)
            if delta is None:
                return False

        self.turn_direction = turn_dir
        self.turn_target_yaw_deg = self._wrap_deg(self.current_yaw_deg + delta)
        self.turn_settle_counter = 0
        self.nav_state = NavState.TURNING
        self.target_yaw_deg = self.turn_target_yaw_deg
        self.prev_heading_error = 0.0
        self.prev_angular_cmd = 0.0

        label = f'{turn_dir.value}' + (' (leg 1/2)' if self._back_turn_leg2_pending else '')
        self.get_logger().info(
            f'Begin turn {label} | current={self.current_yaw_deg:.1f} '
            f'target={self.turn_target_yaw_deg:.1f}')
        return True

    def _execute_turn(self):
        if (not self._gyro_fresh()
                or self.current_yaw_deg is None
                or self.turn_target_yaw_deg is None):
            self._stop()
            return

        err = self._angle_err_deg(self.turn_target_yaw_deg, self.current_yaw_deg)
        abs_err = abs(err)

        if abs_err <= self.turn_tolerance_deg:
            self.turn_settle_counter += 1
        else:
            self.turn_settle_counter = 0

        if self.turn_settle_counter >= self.turn_settle_cycles:
            # If this was the first leg of a BACK turn, start the second 90°
            if self._back_turn_leg2_pending:
                self._back_turn_leg2_pending = False
                second_target = self._wrap_deg(self.current_yaw_deg + 90.0)
                self.turn_target_yaw_deg = second_target
                self.target_yaw_deg = second_target
                self.turn_settle_counter = 0
                self.prev_heading_error = 0.0
                self.prev_angular_cmd = 0.0
                self.get_logger().info(
                    f'BACK leg 2 | current={self.current_yaw_deg:.1f} '
                    f'target={second_target:.1f}')
                return

            self._stop()
            self.nav_state = NavState.FOLLOW
            self.turn_direction = TurnDir.NONE
            self.turn_target_yaw_deg = None
            self.prev_angular_cmd = 0.0
            self.last_junction_time = self._now_sec()
            self.post_turn_forward_cycles = max(
                1, int(self.post_turn_forward_time_sec * self.control_rate_hz))
            self.get_logger().info('Turn complete')
            return

        ang = (self.turn_slow_angular_speed if abs_err < self.turn_slowdown_error_deg
               else self.turn_angular_speed)

        twist = Twist()
        twist.angular.z = ang if err > 0.0 else -ang
        self.cmd_pub.publish(twist)

    # ── junction / cell counting ─────────────────────────────────

    def _publish_counts(self):
        j = Int32(); j.data = self.junction_count
        self.junction_count_pub.publish(j)
        d = Int32(); d.data = self.dead_end_count
        self.dead_end_count_pub.publish(d)

    def _publish_cell_count(self):
        c = Int32(); c.data = self.cell_count
        self.cell_count_pub.publish(c)

    def _reset_distance_segment(self):
        self.reset_distance_pub.publish(Empty())
        self.distance_since_reset_m = 0.0
        self.segment_cells_counted = 0

    def _can_count_event(self):
        return self.distance_since_reset_m >= self.junction_min_distance_m

    def _count_pass_junction(self, left_open, right_open):
        if not self.count_pass_junctions:
            return
        if not self._can_count_event():
            return
        self.junction_count += 1
        self.pass_junction_count += 1
        self._publish_counts()
        self.get_logger().info(
            f'[JUNCTION_PASS] total={self.junction_count} '
            f'pass={self.pass_junction_count} dead={self.dead_end_count} '
            f'dist={self.distance_since_reset_m:.3f}  L={left_open} R={right_open}')
        self._reset_distance_segment()

    def _count_block_or_dead_end(self, left_open, right_open):
        if not self.pending_block_event:
            return
        if not self._can_count_event():
            self.pending_block_event = False
            return

        if left_open or right_open:
            if self.count_blocked_junctions:
                self.junction_count += 1
                self.blocked_junction_count += 1
                self._publish_counts()
                self.get_logger().info(
                    f'[JUNCTION_BLOCK] total={self.junction_count} '
                    f'blocked={self.blocked_junction_count} '
                    f'dist={self.distance_since_reset_m:.3f}  L={left_open} R={right_open}')
        else:
            if self.count_dead_ends:
                self.dead_end_count += 1
                self._publish_counts()
                self.get_logger().info(
                    f'[DEAD_END] junctions={self.junction_count} '
                    f'dead={self.dead_end_count} dist={self.distance_since_reset_m:.3f}')

        self._reset_distance_segment()
        self.pending_block_event = False

    def _update_cell_count(self):
        if self.cell_length_m <= 0.0:
            return
        cells_now = int(self.distance_since_reset_m / self.cell_length_m)
        if cells_now > self.segment_cells_counted:
            new = cells_now - self.segment_cells_counted
            self.segment_cells_counted = cells_now
            self.cell_count += new
            self._publish_cell_count()
            for _ in range(new):
                self._advance_maze_pos()
            self.get_logger().info(
                f'[CELL] total={self.cell_count} seg={self.segment_cells_counted} '
                f'dist={self.distance_since_reset_m:.3f}')

    # ── maze memory ──────────────────────────────────────────────

    def _snap_cardinal(self):
        if self.current_yaw_deg is None or self.initial_gyro_deg is None:
            return None
        rel = self._wrap_deg(
            self.current_yaw_deg - self.initial_gyro_deg + self.maze_heading_offset)
        return self._wrap_deg(round(rel / 90.0) * 90.0)

    @staticmethod
    def _cardinal_delta(cardinal):
        if cardinal is None:
            return (0, 0)
        c = round(cardinal)
        if c == 0:   return (0, 1)    # E
        if c == 90:  return (-1, 0)   # N
        if c == -90: return (1, 0)    # S
        return (0, -1)                 # W

    def _advance_maze_pos(self):
        dr, dc = self._cardinal_delta(self._snap_cardinal())
        nr, nc = self.robot_row + dr, self.robot_col + dc
        if 0 <= nr < self.maze_rows and 0 <= nc < self.maze_cols:
            self.robot_row, self.robot_col = nr, nc
        self.visited.add((self.robot_row, self.robot_col))
        self.get_logger().info(
            f'[MAZE] pos=({self.robot_row},{self.robot_col}) '
            f'heading={self._snap_cardinal()} '
            f'visited={len(self.visited)}/{self.maze_rows * self.maze_cols}')

    def _neighbor_cell(self, turn_dir):
        cardinal = self._snap_cardinal()
        if cardinal is None:
            return None
        offsets = {TurnDir.LEFT: 90.0, TurnDir.RIGHT: -90.0, TurnDir.BACK: 180.0}
        new_h = self._wrap_deg(cardinal + offsets.get(turn_dir, 0.0))
        dr, dc = self._cardinal_delta(new_h)
        return (self.robot_row + dr, self.robot_col + dc)

    def _visited_or_oob(self, cell):
        if cell is None:
            return True
        r, c = cell
        if not (0 <= r < self.maze_rows and 0 <= c < self.maze_cols):
            return True
        return (r, c) in self.visited

    # ── decision logic ───────────────────────────────────────────

    def _begin_refresh(self):
        self._dbg(f'[REFRESH_BEGIN] front={self._fmt(self.front_range)}')
        self.pending_block_event = True
        self.pending_block_event_distance = self.distance_since_reset_m

        # release edge locks & discard stale side data
        self.left_edge_locked = False
        self.right_edge_locked = False
        self.left_range = None
        self.right_range = None
        self.left_valid = False
        self.right_valid = False
        self.left_wall_present = False
        self.right_wall_present = False
        self.left_loss_counter = 0
        self.right_loss_counter = 0

        # reset decision accumulators
        self.left_decision_range = None
        self.right_decision_range = None
        self.left_decision_samples = 0
        self.right_decision_samples = 0

        self.nav_state = NavState.REFRESH_FOR_DECISION

    def _refresh_step(self):
        self._stop()
        if (self.left_decision_samples >= self.decision_required_samples
                and self.right_decision_samples >= self.decision_required_samples):
            self.nav_state = NavState.STOP_AND_DECIDE

    def _choose_turn(self):
        left_open = (self.left_valid and self.left_range is not None
                     and self.left_range > self.turn_open_distance)
        right_open = (self.right_valid and self.right_range is not None
                      and self.right_range > self.turn_open_distance)

        self._count_block_or_dead_end(left_open, right_open)

        self._dbg(
            f'[DECIDE] front={self._fmt(self.front_range)} '
            f'L={self._fmt(self.left_range)} Lopen={left_open} '
            f'R={self._fmt(self.right_range)} Ropen={right_open}')

        if left_open and right_open:
            lv = self._visited_or_oob(self._neighbor_cell(TurnDir.LEFT))
            rv = self._visited_or_oob(self._neighbor_cell(TurnDir.RIGHT))
            if lv and not rv:
                choice = TurnDir.RIGHT
            elif rv and not lv:
                choice = TurnDir.LEFT
            else:
                choice = TurnDir.LEFT if self.prefer_left_first else TurnDir.RIGHT
            self.get_logger().info(
                f'Both open | L={self.left_range:.3f} vis={lv}  '
                f'R={self.right_range:.3f} vis={rv} -> {choice.value}')
            return choice

        if left_open:
            self.get_logger().info(f'Left open ({self.left_range:.3f}) -> LEFT')
            return TurnDir.LEFT
        if right_open:
            self.get_logger().info(f'Right open ({self.right_range:.3f}) -> RIGHT')
            return TurnDir.RIGHT

        self.get_logger().info(
            f'No side open L={self._fmt(self.left_range)} R={self._fmt(self.right_range)} -> BACK')
        return TurnDir.BACK

    # ── apriltag + blink ─────────────────────────────────────────

    def _apriltag_cb(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        incoming = set(payload.get('tag_ids', []))
        new = incoming - self.seen_tag_ids
        if not new:
            return
        self.seen_tag_ids |= new
        decoded = payload.get('decoded_by_order', {})
        self.get_logger().info(
            f'[APRILTAG] New: {sorted(new)} | '
            f'total={len(self.seen_tag_ids)}/8 | decoded={decoded}')
        if self.nav_state == NavState.FOLLOW:
            self._start_blink()
        else:
            self._tag_blink_queued = True

    def _start_blink(self):
        if self._blink_timer is not None:
            self._blink_timer.cancel()
            self._blink_timer = None
        self._blink_count = 0
        self.nav_state = NavState.TAG_BLINK
        self._set_led(True)
        self._blink_timer = self.create_timer(self.blink_half_period_sec, self._blink_step)
        self.get_logger().info('AprilTag detected — stopped, blinking red LED ×2.')

    def _blink_step(self):
        self._blink_count += 1
        self._set_led(self._blink_count % 2 == 0)
        if self._blink_count >= 3:
            self._set_led(False)
            self._blink_timer.cancel()
            self._blink_timer = None
            self.nav_state = NavState.FOLLOW
            self.get_logger().info('Blink complete — resuming.')

    # ── follow controller ────────────────────────────────────────

    def _follow(self):
        wall_term = self._compute_wall_term()
        ff_term = self._compute_ff_term()
        self._update_target_heading(self.mode)
        heading_term = self._compute_heading_term(self.mode)

        if self.ff_hold_counter > 0:
            heading_term *= self.ff_heading_suppress_gain

        wall_w = 0.0
        if self.mode == 'BOTH_WALLS':
            wall_w = self.wall_weight_both
            if (self.front_valid and self.front_range is not None
                    and self.front_range < self.near_front_fusion_distance):
                wall_w = self.wall_weight_near_front

        raw = self._clamp(wall_w * wall_term + heading_term + ff_term,
                          -self.max_angular, self.max_angular)
        raw = self._deadband(raw, self.angular_deadband)
        ang = self._slew(raw, self.prev_angular_cmd, self.angular_slew_per_cycle)
        ang = self._clamp(ang, -self.max_angular, self.max_angular)
        self.prev_angular_cmd = ang

        speed = self._compute_forward_speed(self.mode)

        if self.post_turn_forward_cycles > 0:
            self.post_turn_forward_cycles -= 1
            speed = min(speed, 0.06)

        if speed <= 0.0:
            if not self.stopped:
                self.stopped = True
                d = self.front_range if self.front_range is not None else -1.0
                self.get_logger().info(f'Stopped at front distance: {d:.3f} m')
            self._stop()
            return True

        self.stopped = False
        steer_scale = 1.0 - 0.30 * min(1.0, abs(ang) / max(1e-6, self.max_angular))
        speed = max(self.min_forward_speed, speed * steer_scale)

        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = ang
        self.cmd_pub.publish(twist)
        return False

    # ── main control loop ────────────────────────────────────────

    def _control_loop(self):
        self.debug_cycle_counter += 1
        self._update_cell_count()

        if not (self.left_valid or self.front_valid or self.right_valid or self._gyro_fresh()):
            self._stop()
            return

        if self.nav_state == NavState.TURNING:
            self._execute_turn()
            return

        if self.nav_state == NavState.REFRESH_FOR_DECISION:
            self._refresh_step()
            return

        if self.nav_state == NavState.STOP_AND_DECIDE:
            self._stop()
            self._begin_turn(self._choose_turn())
            return

        if self.nav_state == NavState.TAG_BLINK:
            self._stop()
            return

        # — FOLLOW —
        if self._tag_blink_queued:
            self._tag_blink_queued = False
            self._start_blink()
            return

        self._update_wall_presence()

        left_open = (self.left_valid and self.left_range is not None
                     and self.left_range > self.turn_open_distance)
        right_open = (self.right_valid and self.right_range is not None
                      and self.right_range > self.turn_open_distance)
        front_open = (self.front_valid and self.front_range is not None
                      and self.front_range > self.front_stop_distance + self.front_stop_hysteresis)

        if front_open and (left_open or right_open) and not self._recent_junction():
            self._count_pass_junction(left_open, right_open)
            self.last_junction_time = self._now_sec()

        self._dbg_periodic(
            f'[FOLLOW] F={self._fmt(self.front_range)} '
            f'L={self._fmt(self.left_range)} lockL={self.left_edge_locked} wallL={self.left_wall_present} '
            f'R={self._fmt(self.right_range)} lockR={self.right_edge_locked} wallR={self.right_wall_present} '
            f'mode={self.mode} dist={self.distance_since_reset_m:.3f} cells={self.cell_count}')

        stopped = self._follow()
        if stopped and not self._recent_junction():
            self._dbg('[CONTROL] front stop -> refresh for decision')
            self._begin_refresh()

    # ── shutdown ─────────────────────────────────────────────────

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


# ── entry point ──────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = Task1MazeNode()
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
