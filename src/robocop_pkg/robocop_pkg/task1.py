#!/usr/bin/env python3

import math
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Int32, Empty


class NavState(Enum):
    FOLLOW = 0
    REFRESH_FOR_DECISION = 1
    STOP_AND_DECIDE = 2
    TURNING = 3
    RETURN_TO_JUNCTION = 4
    ENTER_LEFT_JUNCTION = 5


class TurnDir(Enum):
    LEFT = 'L'
    RIGHT = 'R'
    BACK = 'B'
    NONE = 'N'


class Task1MazeNode(Node):
    def __init__(self):
        super().__init__('task1')

        # =========================
        # Topics
        # =========================
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

        # =========================
        # Rates
        # =========================
        self.declare_parameter('control_rate_hz', 20.0)

        # =========================
        # Speeds
        # =========================
        self.declare_parameter('max_forward_speed', 0.10)
        self.declare_parameter('transition_forward_speed', 0.08)
        self.declare_parameter('blind_forward_speed', 0.06)
        self.declare_parameter('min_forward_speed', 0.03)

        # =========================
        # Front stop / slowdown
        # =========================
        self.declare_parameter('front_stop_distance', 0.25)
        self.declare_parameter('front_slow_distance', 0.35)
        self.declare_parameter('front_hard_slow_distance', 0.30)
        self.declare_parameter('front_stop_hysteresis', 0.01)

        # =========================
        # Wall handling
        # =========================
        self.declare_parameter('wall_detect_distance', 0.30)
        self.declare_parameter('wall_clear_distance', 0.35)
        self.declare_parameter('turn_open_distance', 0.28)
        self.declare_parameter('min_valid_range', 0.04)
        self.declare_parameter('max_valid_range', 16.0)
        self.declare_parameter('wall_loss_confirm_cycles', 5)

        # =========================
        # Sensor calibration
        # =========================
        self.declare_parameter('left_sensor_offset_m', -0.004)
        self.declare_parameter('right_sensor_offset_m', 0.004)

        # =========================
        # Edge lock recovery
        # =========================
        self.declare_parameter('edge_reacquire_distance', 0.28)
        self.declare_parameter('edge_reacquire_cycles', 3)
        self.declare_parameter('edge_reacquire_jump', 0.08)

        # =========================
        # Wall control gains
        # =========================
        self.declare_parameter('center_kp', 2.2)
        self.declare_parameter('center_kd', 0.35)

        # =========================
        # Gyro heading-hold gains
        # =========================
        self.declare_parameter('heading_kp', 0.030)
        self.declare_parameter('heading_kd', 0.010)
        self.declare_parameter('heading_weight_both', 0.25)
        self.declare_parameter('heading_weight_missing', 1.00)
        self.declare_parameter('heading_capture_alpha', 0.12)
        self.declare_parameter('heading_error_limit_deg', 25.0)
        self.declare_parameter('gyro_valid_timeout_sec', 0.50)

        # =========================
        # Linear motion mode options
        # =========================
        self.declare_parameter('use_gyro_fuse_linear', True)
        self.declare_parameter('use_single_wall_linear', False)
        self.declare_parameter('single_wall_target_distance', 0.15)

        # =========================
        # Fusion shaping
        # =========================
        self.declare_parameter('wall_weight_both', 1.00)
        self.declare_parameter('wall_weight_near_front', 0.35)
        self.declare_parameter('near_front_fusion_distance', 0.30)

        # =========================
        # Output shaping
        # =========================
        self.declare_parameter('max_angular', 0.8)
        self.declare_parameter('angular_deadband', 0.01)
        self.declare_parameter('angular_slew_per_cycle', 0.10)

        # =========================
        # Filtering / edge handling
        # =========================
        self.declare_parameter('side_filter_alpha', 0.22)
        self.declare_parameter('front_filter_alpha', 0.30)
        self.declare_parameter('max_side_jump', 0.14)
        self.declare_parameter('max_front_jump', 0.22)
        self.declare_parameter('corridor_width_alpha', 0.10)

        # =========================
        # Decision refresh
        # =========================
        self.declare_parameter('decision_required_samples', 5)
        self.declare_parameter('decision_filter_alpha', 0.35)

        # =========================
        # Feed-forward wall escape
        # =========================
        self.declare_parameter('ff_enabled', False)
        self.declare_parameter('ff_trigger_delta', 0.002)
        self.declare_parameter('ff_trigger_cycles', 2)
        self.declare_parameter('ff_turn_mag', 0.12)
        self.declare_parameter('ff_hold_cycles', 4)
        self.declare_parameter('ff_cooldown_cycles', 5)
        self.declare_parameter('ff_front_min_distance', 0.10)
        self.declare_parameter('ff_edge_lock_block', False)
        self.declare_parameter('ff_heading_suppress_gain', 0.4)

        self.declare_parameter('ff_enable_both_walls', True)
        self.declare_parameter('ff_both_trigger_error_m', 0.006)
        self.declare_parameter('ff_both_trigger_cycles', 2)
        self.declare_parameter('ff_both_turn_mag', 0.10)

        # =========================
        # Junction counting
        # =========================
        self.declare_parameter('junction_min_distance_m', 0.12)
        self.declare_parameter('count_pass_junctions', True)
        self.declare_parameter('count_blocked_junctions', True)
        self.declare_parameter('count_dead_ends', True)

        # =========================
        # Cell counting
        # =========================
        self.declare_parameter('cell_length_m', 0.40)

        # =========================
        # Debug
        # =========================
        self.declare_parameter('debug_logs', True)
        self.declare_parameter('debug_every_n_cycles', 5)

        # =========================
        # Turn controller
        # =========================
        self.declare_parameter('turn_angular_speed', 0.45)
        self.declare_parameter('turn_slow_angular_speed', 0.22)
        self.declare_parameter('turn_slowdown_error_deg', 18.0)
        self.declare_parameter('turn_tolerance_deg', 3.0)
        self.declare_parameter('turn_settle_cycles', 4)
        self.declare_parameter('post_turn_forward_time_sec', 0.35)
        self.declare_parameter('junction_cooldown_sec', 0.80)
        self.declare_parameter('prefer_left_first', True)

        # =========================
        # Enter junction before left turn
        # =========================
        self.declare_parameter('left_entry_distance_m', 0.20)
        self.declare_parameter('left_entry_speed', 0.06)

        # =========================
        # Read params
        # =========================
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.left_range_topic = self.get_parameter('left_range_topic').value
        self.front_range_topic = self.get_parameter('front_range_topic').value
        self.right_range_topic = self.get_parameter('right_range_topic').value
        self.gyro_angle_topic = self.get_parameter('gyro_angle_topic').value

        self.distance_since_reset_topic = self.get_parameter('distance_since_reset_topic').value
        self.reset_distance_topic = self.get_parameter('reset_distance_topic').value
        self.junction_count_topic = self.get_parameter('junction_count_topic').value
        self.dead_end_count_topic = self.get_parameter('dead_end_count_topic').value
        self.cell_count_topic = self.get_parameter('cell_count_topic').value

        self.control_rate_hz = float(self.get_parameter('control_rate_hz').value)

        self.max_forward_speed = float(self.get_parameter('max_forward_speed').value)
        self.transition_forward_speed = float(self.get_parameter('transition_forward_speed').value)
        self.blind_forward_speed = float(self.get_parameter('blind_forward_speed').value)
        self.min_forward_speed = float(self.get_parameter('min_forward_speed').value)

        self.front_stop_distance = float(self.get_parameter('front_stop_distance').value)
        self.front_slow_distance = float(self.get_parameter('front_slow_distance').value)
        self.front_hard_slow_distance = float(self.get_parameter('front_hard_slow_distance').value)
        self.front_stop_hysteresis = float(self.get_parameter('front_stop_hysteresis').value)

        self.wall_detect_distance = float(self.get_parameter('wall_detect_distance').value)
        self.wall_clear_distance = float(self.get_parameter('wall_clear_distance').value)
        self.turn_open_distance = float(self.get_parameter('turn_open_distance').value)
        self.min_valid_range = float(self.get_parameter('min_valid_range').value)
        self.max_valid_range = float(self.get_parameter('max_valid_range').value)
        self.wall_loss_confirm_cycles = int(self.get_parameter('wall_loss_confirm_cycles').value)

        self.left_sensor_offset_m = float(self.get_parameter('left_sensor_offset_m').value)
        self.right_sensor_offset_m = float(self.get_parameter('right_sensor_offset_m').value)

        self.edge_reacquire_distance = float(self.get_parameter('edge_reacquire_distance').value)
        self.edge_reacquire_cycles = int(self.get_parameter('edge_reacquire_cycles').value)
        self.edge_reacquire_jump = float(self.get_parameter('edge_reacquire_jump').value)

        self.center_kp = float(self.get_parameter('center_kp').value)
        self.center_kd = float(self.get_parameter('center_kd').value)

        self.heading_kp = float(self.get_parameter('heading_kp').value)
        self.heading_kd = float(self.get_parameter('heading_kd').value)
        self.heading_weight_both = float(self.get_parameter('heading_weight_both').value)
        self.heading_weight_missing = float(self.get_parameter('heading_weight_missing').value)
        self.heading_capture_alpha = float(self.get_parameter('heading_capture_alpha').value)
        self.heading_error_limit_deg = float(self.get_parameter('heading_error_limit_deg').value)
        self.gyro_valid_timeout_sec = float(self.get_parameter('gyro_valid_timeout_sec').value)

        self.use_gyro_fuse_linear = bool(self.get_parameter('use_gyro_fuse_linear').value)
        self.use_single_wall_linear = bool(self.get_parameter('use_single_wall_linear').value)
        self.single_wall_target_distance = float(self.get_parameter('single_wall_target_distance').value)

        self.wall_weight_both = float(self.get_parameter('wall_weight_both').value)
        self.wall_weight_near_front = float(self.get_parameter('wall_weight_near_front').value)
        self.near_front_fusion_distance = float(self.get_parameter('near_front_fusion_distance').value)

        self.max_angular = float(self.get_parameter('max_angular').value)
        self.angular_deadband = float(self.get_parameter('angular_deadband').value)
        self.angular_slew_per_cycle = float(self.get_parameter('angular_slew_per_cycle').value)

        self.side_filter_alpha = float(self.get_parameter('side_filter_alpha').value)
        self.front_filter_alpha = float(self.get_parameter('front_filter_alpha').value)
        self.max_side_jump = float(self.get_parameter('max_side_jump').value)
        self.max_front_jump = float(self.get_parameter('max_front_jump').value)
        self.corridor_width_alpha = float(self.get_parameter('corridor_width_alpha').value)

        self.decision_required_samples = int(self.get_parameter('decision_required_samples').value)
        self.decision_filter_alpha = float(self.get_parameter('decision_filter_alpha').value)

        self.ff_enabled = bool(self.get_parameter('ff_enabled').value)
        self.ff_trigger_delta = float(self.get_parameter('ff_trigger_delta').value)
        self.ff_trigger_cycles = int(self.get_parameter('ff_trigger_cycles').value)
        self.ff_turn_mag = float(self.get_parameter('ff_turn_mag').value)
        self.ff_hold_cycles = int(self.get_parameter('ff_hold_cycles').value)
        self.ff_cooldown_cycles = int(self.get_parameter('ff_cooldown_cycles').value)
        self.ff_front_min_distance = float(self.get_parameter('ff_front_min_distance').value)
        self.ff_edge_lock_block = bool(self.get_parameter('ff_edge_lock_block').value)
        self.ff_heading_suppress_gain = float(self.get_parameter('ff_heading_suppress_gain').value)

        self.ff_enable_both_walls = bool(self.get_parameter('ff_enable_both_walls').value)
        self.ff_both_trigger_error_m = float(self.get_parameter('ff_both_trigger_error_m').value)
        self.ff_both_trigger_cycles = int(self.get_parameter('ff_both_trigger_cycles').value)
        self.ff_both_turn_mag = float(self.get_parameter('ff_both_turn_mag').value)

        self.junction_min_distance_m = float(self.get_parameter('junction_min_distance_m').value)
        self.count_pass_junctions = bool(self.get_parameter('count_pass_junctions').value)
        self.count_blocked_junctions = bool(self.get_parameter('count_blocked_junctions').value)
        self.count_dead_ends = bool(self.get_parameter('count_dead_ends').value)

        self.cell_length_m = float(self.get_parameter('cell_length_m').value)

        self.debug_logs = bool(self.get_parameter('debug_logs').value)
        self.debug_every_n_cycles = int(self.get_parameter('debug_every_n_cycles').value)

        self.turn_angular_speed = float(self.get_parameter('turn_angular_speed').value)
        self.turn_slow_angular_speed = float(self.get_parameter('turn_slow_angular_speed').value)
        self.turn_slowdown_error_deg = float(self.get_parameter('turn_slowdown_error_deg').value)
        self.turn_tolerance_deg = float(self.get_parameter('turn_tolerance_deg').value)
        self.turn_settle_cycles = int(self.get_parameter('turn_settle_cycles').value)
        self.post_turn_forward_time_sec = float(self.get_parameter('post_turn_forward_time_sec').value)
        self.junction_cooldown_sec = float(self.get_parameter('junction_cooldown_sec').value)
        self.prefer_left_first = bool(self.get_parameter('prefer_left_first').value)

        self.left_entry_distance_m = float(self.get_parameter('left_entry_distance_m').value)
        self.left_entry_speed = float(self.get_parameter('left_entry_speed').value)

        # =========================
        # State
        # =========================
        self.left_range = None
        self.front_range = None
        self.right_range = None

        self.left_valid = False
        self.front_valid = False
        self.right_valid = False

        self.left_wall_present = False
        self.right_wall_present = False

        self.left_loss_counter = 0
        self.right_loss_counter = 0

        self.left_edge_locked = False
        self.right_edge_locked = False
        self.left_reacquire_counter = 0
        self.right_reacquire_counter = 0

        self.prev_center_error = 0.0
        self.prev_heading_error = 0.0
        self.prev_angular_cmd = 0.0
        self.last_stable_angular = 0.0

        self.corridor_width_est = None
        self.mode = 'INIT'
        self.stopped = False

        self.current_yaw_deg = None
        self.last_yaw_deg = None
        self.target_yaw_deg = None
        self.last_gyro_msg_time = None
        self.last_linear_mode = None

        self.nav_state = NavState.FOLLOW
        self.turn_direction = TurnDir.NONE
        self.turn_target_yaw_deg = None
        self.turn_settle_counter = 0
        self.post_turn_forward_cycles = 0
        self.last_junction_time = None

        self.left_decision_range = None
        self.right_decision_range = None
        self.left_decision_samples = 0
        self.right_decision_samples = 0

        self.prev_left_ff_range = None
        self.prev_right_ff_range = None
        self.left_toward_wall_count = 0
        self.right_toward_wall_count = 0
        self.ff_active_dir = TurnDir.NONE
        self.ff_hold_counter = 0
        self.ff_cooldown_counter = 0
        self.ff_both_left_count = 0
        self.ff_both_right_count = 0
        self.ff_current_mag = 0.0

        self.debug_cycle_counter = 0
        self.last_left_raw = None
        self.last_right_raw = None
        self.last_front_raw = None

        self.distance_since_reset_m = 0.0
        self.junction_count = 0
        self.dead_end_count = 0
        self.pass_junction_count = 0
        self.blocked_junction_count = 0
        self.pending_block_event = False
        self.pending_block_event_distance = 0.0

        self.cell_count = 0
        self.segment_cells_counted = 0

        self.return_after_back_turn = False
        self.return_distance_target_m = 0.0
        self.return_stop_margin_m = 0.05

        self.last_junction_heading_deg = None
        self.last_junction_front_open = False
        self.last_junction_left_open = False
        self.last_junction_right_open = False
        self.last_junction_exit_heading_deg = None

        self.left_entry_target_distance_m = 0.0
        self.left_entry_start_distance_m = 0.0

        # ROS
        self.left_sub = self.create_subscription(
            Range, self.left_range_topic, self.left_cb, qos_profile_sensor_data
        )
        self.front_sub = self.create_subscription(
            Range, self.front_range_topic, self.front_cb, qos_profile_sensor_data
        )
        self.right_sub = self.create_subscription(
            Range, self.right_range_topic, self.right_cb, qos_profile_sensor_data
        )
        self.gyro_sub = self.create_subscription(
            Float32, self.gyro_angle_topic, self.gyro_cb, qos_profile_sensor_data
        )
        self.distance_sub = self.create_subscription(
            Float32, self.distance_since_reset_topic, self.distance_since_reset_cb, 10
        )

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.reset_distance_pub = self.create_publisher(Empty, self.reset_distance_topic, 10)
        self.junction_count_pub = self.create_publisher(Int32, self.junction_count_topic, 10)
        self.dead_end_count_pub = self.create_publisher(Int32, self.dead_end_count_topic, 10)
        self.cell_count_pub = self.create_publisher(Int32, self.cell_count_topic, 10)

        self.timer = self.create_timer(1.0 / self.control_rate_hz, self.control_loop)

        self.get_logger().info('Task1 maze node started')
        self.get_logger().info(
            f'left={self.left_range_topic}, front={self.front_range_topic}, '
            f'right={self.right_range_topic}, gyro={self.gyro_angle_topic}'
        )

    @staticmethod
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    @staticmethod
    def wrap_angle_deg(angle_deg):
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0
        return angle_deg

    def angle_error_deg(self, target_deg, current_deg):
        return self.wrap_angle_deg(target_deg - current_deg)

    def is_valid_measurement(self, x):
        return (
            x is not None and
            not math.isnan(x) and
            not math.isinf(x) and
            self.min_valid_range <= x <= self.max_valid_range
        )

    def ema_update(self, old_val, new_val, alpha):
        if old_val is None:
            return new_val
        return alpha * new_val + (1.0 - alpha) * old_val

    def apply_deadband(self, x, db):
        if abs(x) < db:
            return 0.0
        return x

    def slew_limit(self, target, previous, max_step):
        if target > previous + max_step:
            return previous + max_step
        if target < previous - max_step:
            return previous - max_step
        return target

    def gyro_is_fresh(self):
        if self.last_gyro_msg_time is None:
            return False
        dt = (self.get_clock().now() - self.last_gyro_msg_time).nanoseconds * 1e-9
        return dt <= self.gyro_valid_timeout_sec

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def stop_cmd(self):
        self.cmd_pub.publish(Twist())

    def fmt(self, x):
        if x is None:
            return 'None'
        return f'{x:.3f}'

    def debug_print(self, msg):
        if self.debug_logs:
            self.get_logger().info(msg)

    def debug_periodic(self, msg):
        if not self.debug_logs:
            return
        if self.debug_every_n_cycles <= 1:
            self.get_logger().info(msg)
            return
        if (self.debug_cycle_counter % self.debug_every_n_cycles) == 0:
            self.get_logger().info(msg)

    def corrected_left(self):
        if self.left_range is None:
            return None
        return self.left_range + self.left_sensor_offset_m

    def corrected_right(self):
        if self.right_range is None:
            return None
        return self.right_range + self.right_sensor_offset_m

    def capture_heading_reference(self, reason=''):
        if not self.use_gyro_fuse_linear:
            return
        if not self.gyro_is_fresh() or self.current_yaw_deg is None:
            return
        self.target_yaw_deg = self.current_yaw_deg
        self.prev_heading_error = 0.0
        if reason:
            self.debug_print(f'[HEADING_CAPTURE] reason={reason} yaw={self.current_yaw_deg:.2f}')

    def heading_for_turn(self, base_heading_deg, turn_dir: TurnDir):
        if turn_dir == TurnDir.LEFT:
            return self.wrap_angle_deg(base_heading_deg + 90.0)
        if turn_dir == TurnDir.RIGHT:
            return self.wrap_angle_deg(base_heading_deg - 90.0)
        if turn_dir == TurnDir.BACK:
            return self.wrap_angle_deg(base_heading_deg + 180.0)
        return self.wrap_angle_deg(base_heading_deg)

    def remember_junction(self, front_open: bool, left_open: bool, right_open: bool, chosen_dir: TurnDir):
        if not self.gyro_is_fresh() or self.current_yaw_deg is None:
            return

        base = self.current_yaw_deg
        self.last_junction_heading_deg = base
        self.last_junction_front_open = front_open
        self.last_junction_left_open = left_open
        self.last_junction_right_open = right_open
        self.last_junction_exit_heading_deg = self.heading_for_turn(base, chosen_dir)

    def choose_return_target_heading(self):
        if self.last_junction_heading_deg is None or self.last_junction_exit_heading_deg is None:
            return None

        base = self.last_junction_heading_deg
        came_from_branch = self.last_junction_exit_heading_deg
        incoming_heading = self.wrap_angle_deg(base + 180.0)

        candidates = []
        if self.last_junction_left_open:
            candidates.append(('LEFT', self.wrap_angle_deg(base + 90.0)))
        if self.last_junction_front_open:
            candidates.append(('FRONT', self.wrap_angle_deg(base)))
        if self.last_junction_right_open:
            candidates.append(('RIGHT', self.wrap_angle_deg(base - 90.0)))

        candidates = [
            (name, hdg) for name, hdg in candidates
            if abs(self.angle_error_deg(came_from_branch, hdg)) > 20.0
        ]

        if not candidates:
            candidates = [('IN', incoming_heading)]

        order = ['LEFT', 'FRONT', 'RIGHT', 'IN'] if self.prefer_left_first else ['RIGHT', 'FRONT', 'LEFT', 'IN']
        candidates.sort(key=lambda item: order.index(item[0]))
        return candidates[0][1]

    def heading_to_turn_dir(self, target_heading_deg):
        if not self.gyro_is_fresh() or self.current_yaw_deg is None or target_heading_deg is None:
            return TurnDir.NONE

        err = self.angle_error_deg(target_heading_deg, self.current_yaw_deg)

        if abs(err) <= self.turn_tolerance_deg:
            return TurnDir.NONE
        if 45.0 <= err <= 135.0:
            return TurnDir.LEFT
        if -135.0 <= err <= -45.0:
            return TurnDir.RIGHT
        return TurnDir.BACK

    def side_open_from_value(self, value):
        return (
            value is not None and
            self.is_valid_measurement(value) and
            value > self.turn_open_distance
        )

    def left_open_for_turn(self):
        if self.side_open_from_value(self.last_left_raw):
            return True
        return (
            self.left_valid and
            self.left_range is not None and
            self.left_range > self.turn_open_distance
        )

    def right_open_for_turn(self):
        if self.side_open_from_value(self.last_right_raw):
            return True
        return (
            self.right_valid and
            self.right_range is not None and
            self.right_range > self.turn_open_distance
        )

    def front_open_for_turn(self):
        return (
            self.front_valid and
            self.front_range is not None and
            self.front_range > (self.front_stop_distance + self.front_stop_hysteresis)
        )

    def distance_since_reset_cb(self, msg: Float32):
        self.distance_since_reset_m = float(msg.data)

    def publish_counts(self):
        j = Int32()
        j.data = int(self.junction_count)
        self.junction_count_pub.publish(j)

        d = Int32()
        d.data = int(self.dead_end_count)
        self.dead_end_count_pub.publish(d)

    def publish_cell_count(self):
        c = Int32()
        c.data = int(self.cell_count)
        self.cell_count_pub.publish(c)

    def update_cell_count(self):
        if self.cell_length_m <= 0.0:
            return
        cells_now = int(self.distance_since_reset_m / self.cell_length_m)
        if cells_now > self.segment_cells_counted:
            new_cells = cells_now - self.segment_cells_counted
            self.segment_cells_counted = cells_now
            self.cell_count += new_cells
            self.publish_cell_count()

    def reset_distance_segment(self):
        self.reset_distance_pub.publish(Empty())
        self.distance_since_reset_m = 0.0
        self.segment_cells_counted = 0

    def can_count_new_event(self):
        return self.distance_since_reset_m >= self.junction_min_distance_m

    def count_pass_junction_event(self, left_open: bool, right_open: bool):
        if not self.count_pass_junctions:
            return
        if not self.can_count_new_event():
            return
        self.junction_count += 1
        self.pass_junction_count += 1
        self.publish_counts()
        self.reset_distance_segment()

    def count_block_or_dead_end_event(self, left_open: bool, right_open: bool):
        if not self.pending_block_event:
            return

        if not self.can_count_new_event():
            self.pending_block_event = False
            return

        if left_open or right_open:
            if self.count_blocked_junctions:
                self.junction_count += 1
                self.blocked_junction_count += 1
                self.publish_counts()
            self.reset_distance_segment()

        self.pending_block_event = False

    def begin_refresh_for_decision(self):
        self.pending_block_event = True
        self.pending_block_event_distance = self.distance_since_reset_m

        self.left_edge_locked = False
        self.right_edge_locked = False
        self.left_reacquire_counter = 0
        self.right_reacquire_counter = 0

        self.left_range = None
        self.right_range = None
        self.left_valid = False
        self.right_valid = False

        self.left_wall_present = False
        self.right_wall_present = False
        self.left_loss_counter = 0
        self.right_loss_counter = 0

        self.left_decision_range = None
        self.right_decision_range = None
        self.left_decision_samples = 0
        self.right_decision_samples = 0

        self.nav_state = NavState.REFRESH_FOR_DECISION

    def left_ready_for_decision(self):
        return self.left_decision_samples >= self.decision_required_samples

    def right_ready_for_decision(self):
        return self.right_decision_samples >= self.decision_required_samples

    def both_ready_for_decision(self):
        return self.left_ready_for_decision() and self.right_ready_for_decision()

    def update_wall_presence(self):
        if self.left_valid:
            if self.left_wall_present:
                if self.left_range > self.wall_clear_distance:
                    self.left_loss_counter += 1
                    if self.left_loss_counter >= self.wall_loss_confirm_cycles:
                        self.left_wall_present = False
                        self.left_loss_counter = 0
                else:
                    self.left_loss_counter = 0
            else:
                if self.left_range < self.wall_detect_distance:
                    self.left_wall_present = True
                    self.left_loss_counter = 0

        if self.right_valid:
            if self.right_wall_present:
                if self.right_range > self.wall_clear_distance:
                    self.right_loss_counter += 1
                    if self.right_loss_counter >= self.wall_loss_confirm_cycles:
                        self.right_wall_present = False
                        self.right_loss_counter = 0
                else:
                    self.right_loss_counter = 0
            else:
                if self.right_range < self.wall_detect_distance:
                    self.right_wall_present = True
                    self.right_loss_counter = 0

    def handle_left_locked(self, raw_r):
        candidate_close = raw_r < self.edge_reacquire_distance
        candidate_jump_ok = (self.left_range is None) or (abs(raw_r - self.left_range) < self.edge_reacquire_jump)

        if candidate_close or candidate_jump_ok:
            self.left_reacquire_counter += 1
        else:
            self.left_reacquire_counter = 0

        if self.left_reacquire_counter >= self.edge_reacquire_cycles:
            self.left_edge_locked = False
            self.left_reacquire_counter = 0
            self.left_range = raw_r
            self.left_valid = True
            self.left_wall_present = raw_r < self.wall_detect_distance
            self.debug_print(f'[LEFT_UNLOCK] reacquired raw={raw_r:.3f}')

    def handle_right_locked(self, raw_r):
        candidate_close = raw_r < self.edge_reacquire_distance
        candidate_jump_ok = (self.right_range is None) or (abs(raw_r - self.right_range) < self.edge_reacquire_jump)

        if candidate_close or candidate_jump_ok:
            self.right_reacquire_counter += 1
        else:
            self.right_reacquire_counter = 0

        if self.right_reacquire_counter >= self.edge_reacquire_cycles:
            self.right_edge_locked = False
            self.right_reacquire_counter = 0
            self.right_range = raw_r
            self.right_valid = True
            self.right_wall_present = raw_r < self.wall_detect_distance
            self.debug_print(f'[RIGHT_UNLOCK] reacquired raw={raw_r:.3f}')

    def left_cb(self, msg: Range):
        r = float(msg.range)
        self.last_left_raw = r

        if not self.is_valid_measurement(r):
            return

        if self.nav_state == NavState.REFRESH_FOR_DECISION:
            self.left_decision_range = self.ema_update(
                self.left_decision_range, r, self.decision_filter_alpha
            )
            self.left_decision_samples += 1
            self.left_range = self.left_decision_range
            self.left_valid = True
            return

        if self.left_edge_locked:
            self.handle_left_locked(r)
            return

        if self.left_range is None:
            self.left_range = r
        else:
            diff = abs(r - self.left_range)
            if diff > self.max_side_jump:
                self.left_edge_locked = True
                self.left_reacquire_counter = 0
                self.debug_print(f'[LEFT_EDGE_LOCK] raw={r:.3f} cur={self.left_range:.3f} diff={diff:.3f}')
                self.handle_left_locked(r)
                return
            else:
                self.left_range = self.ema_update(
                    self.left_range, r, self.side_filter_alpha
                )

        self.left_valid = True

    def front_cb(self, msg: Range):
        r = float(msg.range)
        self.last_front_raw = r

        if not self.is_valid_measurement(r):
            return

        self.front_range = r
        self.front_valid = True

    def right_cb(self, msg: Range):
        r = float(msg.range)
        self.last_right_raw = r

        if not self.is_valid_measurement(r):
            return

        if self.nav_state == NavState.REFRESH_FOR_DECISION:
            self.right_decision_range = self.ema_update(
                self.right_decision_range, r, self.decision_filter_alpha
            )
            self.right_decision_samples += 1
            self.right_range = self.right_decision_range
            self.right_valid = True
            return

        if self.right_edge_locked:
            self.handle_right_locked(r)
            return

        if self.right_range is None:
            self.right_range = r
        else:
            diff = abs(r - self.right_range)
            if diff > self.max_side_jump:
                self.right_edge_locked = True
                self.right_reacquire_counter = 0
                self.debug_print(f'[RIGHT_EDGE_LOCK] raw={r:.3f} cur={self.right_range:.3f} diff={diff:.3f}')
                self.handle_right_locked(r)
                return
            else:
                self.right_range = self.ema_update(
                    self.right_range, r, self.side_filter_alpha
                )

        self.right_valid = True

    def gyro_cb(self, msg: Float32):
        yaw_deg = float(msg.data)
        yaw_deg = self.wrap_angle_deg(yaw_deg)
        self.last_yaw_deg = self.current_yaw_deg
        self.current_yaw_deg = yaw_deg
        self.last_gyro_msg_time = self.get_clock().now()

        if self.target_yaw_deg is None:
            self.capture_heading_reference('gyro_init')

    def compute_forward_speed(self, mode):
        if not self.front_valid or self.front_range is None:
            if mode in ['GYRO_ONLY', 'OPEN_LOOP']:
                return self.blind_forward_speed
            if mode == 'TRANSITION':
                return self.transition_forward_speed
            return min(0.08, self.max_forward_speed)

        if self.front_range <= (self.front_stop_distance + self.front_stop_hysteresis):
            return 0.0

        if mode in ['GYRO_ONLY', 'OPEN_LOOP']:
            mode_cap = self.blind_forward_speed
        elif mode == 'TRANSITION':
            mode_cap = self.transition_forward_speed
        else:
            mode_cap = self.max_forward_speed

        if self.front_range <= self.front_hard_slow_distance:
            span = max(1e-6, self.front_hard_slow_distance - self.front_stop_distance)
            ratio = (self.front_range - self.front_stop_distance) / span
            ratio = self.clamp(ratio, 0.0, 1.0)
            speed = self.min_forward_speed + ratio * (min(mode_cap, 0.07) - self.min_forward_speed)
            return self.clamp(speed, self.min_forward_speed, mode_cap)

        if self.front_range <= self.front_slow_distance:
            span = max(1e-6, self.front_slow_distance - self.front_hard_slow_distance)
            ratio = (self.front_range - self.front_hard_slow_distance) / span
            ratio = self.clamp(ratio, 0.0, 1.0)
            low = min(mode_cap, 0.07)
            speed = low + ratio * (mode_cap - low)
            return self.clamp(speed, self.min_forward_speed, mode_cap)

        return mode_cap

    def maybe_update_target_heading(self, mode):
        if not self.use_gyro_fuse_linear:
            return
        if not self.gyro_is_fresh() or self.current_yaw_deg is None:
            return

        if self.target_yaw_deg is None:
            self.capture_heading_reference('target_none')

        if self.last_linear_mode != mode:
            if mode in ['LEFT_WALL', 'RIGHT_WALL', 'GYRO_ONLY']:
                self.capture_heading_reference(f'mode_change_{mode}')
            self.last_linear_mode = mode

    def compute_heading_term(self, mode):
        if not self.use_gyro_fuse_linear:
            return 0.0

        if not self.gyro_is_fresh() or self.current_yaw_deg is None or self.target_yaw_deg is None:
            return 0.0

        heading_error_deg = self.angle_error_deg(self.target_yaw_deg, self.current_yaw_deg)
        heading_error_deg = self.clamp(
            heading_error_deg,
            -self.heading_error_limit_deg,
            self.heading_error_limit_deg
        )

        derivative = (heading_error_deg - self.prev_heading_error) * self.control_rate_hz
        self.prev_heading_error = heading_error_deg

        heading_term = self.heading_kp * heading_error_deg + self.heading_kd * derivative

        if mode == 'BOTH_WALLS':
            heading_term *= self.heading_weight_both
        else:
            heading_term *= self.heading_weight_missing

        return heading_term

    def reset_ff_trackers(self):
        self.left_toward_wall_count = 0
        self.right_toward_wall_count = 0
        self.ff_both_left_count = 0
        self.ff_both_right_count = 0
        self.prev_left_ff_range = self.corrected_left()
        self.prev_right_ff_range = self.corrected_right()

    def can_use_ff(self):
        if not self.ff_enabled:
            return False
        if not self.front_valid or self.front_range is None:
            return False
        if self.front_range <= self.ff_front_min_distance:
            return False
        if self.ff_cooldown_counter > 0:
            return False
        return True

    def start_ff_pulse(self, direction: TurnDir, reason: str, mag: float = None):
        self.ff_active_dir = direction
        self.ff_hold_counter = self.ff_hold_cycles
        self.ff_cooldown_counter = self.ff_cooldown_cycles
        self.ff_current_mag = self.ff_turn_mag if mag is None else mag
        self.debug_print(f'[FF_START] dir={direction.value} mag={self.ff_current_mag:.3f} reason={reason}')

    def ff_pulse_active(self):
        return self.ff_hold_counter > 0

    def compute_feedforward_term(self):
        if self.ff_hold_counter > 0:
            self.ff_hold_counter -= 1

            if self.ff_active_dir == TurnDir.RIGHT:
                ang = -self.ff_current_mag
            elif self.ff_active_dir == TurnDir.LEFT:
                ang = self.ff_current_mag
            else:
                ang = 0.0

            if self.ff_hold_counter == 0:
                self.debug_print(f'[FF_END] dir={self.ff_active_dir.value}')
                self.ff_active_dir = TurnDir.NONE

            return ang

        if self.ff_cooldown_counter > 0:
            self.ff_cooldown_counter -= 1

        if not self.can_use_ff():
            self.reset_ff_trackers()
            return 0.0

        left_corr = self.corrected_left()
        right_corr = self.corrected_right()

        left = left_corr is not None and self.left_valid and self.left_range < self.wall_detect_distance
        right = right_corr is not None and self.right_valid and self.right_range < self.wall_detect_distance

        if self.ff_enable_both_walls and left and right:
            error = left_corr - right_corr

            if error > self.ff_both_trigger_error_m:
                self.ff_both_left_count += 1
                self.ff_both_right_count = 0
            elif error < -self.ff_both_trigger_error_m:
                self.ff_both_right_count += 1
                self.ff_both_left_count = 0
            else:
                self.ff_both_left_count = max(0, self.ff_both_left_count - 1)
                self.ff_both_right_count = max(0, self.ff_both_right_count - 1)

            if self.ff_both_left_count >= self.ff_both_trigger_cycles:
                self.ff_both_left_count = 0
                self.start_ff_pulse(TurnDir.RIGHT, 'both_walls_drifting_left', self.ff_both_turn_mag)
                return -self.ff_both_turn_mag

            if self.ff_both_right_count >= self.ff_both_trigger_cycles:
                self.ff_both_right_count = 0
                self.start_ff_pulse(TurnDir.LEFT, 'both_walls_drifting_right', self.ff_both_turn_mag)
                return self.ff_both_turn_mag

            return 0.0

        if left and not right:
            prev = self.prev_left_ff_range
            if prev is not None and left_corr < (prev - self.ff_trigger_delta):
                self.left_toward_wall_count += 1
            else:
                self.left_toward_wall_count = max(0, self.left_toward_wall_count - 1)

            self.prev_left_ff_range = left_corr
            self.right_toward_wall_count = 0

            if self.left_toward_wall_count >= self.ff_trigger_cycles:
                self.left_toward_wall_count = 0
                self.start_ff_pulse(TurnDir.RIGHT, 'pushing_toward_left_wall')
                return -self.ff_turn_mag

        if right and not left:
            prev = self.prev_right_ff_range
            if prev is not None and right_corr < (prev - self.ff_trigger_delta):
                self.right_toward_wall_count += 1
            else:
                self.right_toward_wall_count = max(0, self.right_toward_wall_count - 1)

            self.prev_right_ff_range = right_corr
            self.left_toward_wall_count = 0

            if self.right_toward_wall_count >= self.ff_trigger_cycles:
                self.right_toward_wall_count = 0
                self.start_ff_pulse(TurnDir.LEFT, 'pushing_toward_right_wall')
                return self.ff_turn_mag

        return 0.0

    def compute_wall_term(self):
        left_ready = (
            self.left_valid and
            self.left_range is not None and
            not self.left_edge_locked and
            self.left_range < self.wall_detect_distance
        )
        right_ready = (
            self.right_valid and
            self.right_range is not None and
            not self.right_edge_locked and
            self.right_range < self.wall_detect_distance
        )

        if left_ready and right_ready:
            left_corr = self.corrected_left()
            right_corr = self.corrected_right()

            measured_width = left_corr + right_corr
            if self.corridor_width_est is None:
                self.corridor_width_est = measured_width
            else:
                a = self.corridor_width_alpha
                self.corridor_width_est = a * measured_width + (1.0 - a) * self.corridor_width_est

            if self.mode != 'BOTH_WALLS':
                self.prev_center_error = 0.0
                self.debug_print('[MODE] -> BOTH_WALLS')

            self.mode = 'BOTH_WALLS'
            error = left_corr - right_corr
            derivative = (error - self.prev_center_error) * self.control_rate_hz
            self.prev_center_error = error

            ang = self.center_kp * error + self.center_kd * derivative
            self.last_stable_angular = 0.85 * self.last_stable_angular + 0.15 * ang
            return ang

        if self.use_single_wall_linear:
            if left_ready and not right_ready:
                self.mode = 'LEFT_WALL'
                left_corr = self.corrected_left()
                error = self.single_wall_target_distance - left_corr
                derivative = (error - self.prev_center_error) * self.control_rate_hz
                self.prev_center_error = error

                ang = self.center_kp * error + self.center_kd * derivative
                self.last_stable_angular = 0.85 * self.last_stable_angular + 0.15 * ang
                return ang

            if right_ready and not left_ready:
                self.mode = 'RIGHT_WALL'
                right_corr = self.corrected_right()
                error = right_corr - self.single_wall_target_distance
                derivative = (error - self.prev_center_error) * self.control_rate_hz
                self.prev_center_error = error

                ang = self.center_kp * error + self.center_kd * derivative
                self.last_stable_angular = 0.85 * self.last_stable_angular + 0.15 * ang
                return ang

        self.mode = 'GYRO_ONLY' if self.use_gyro_fuse_linear else 'OPEN_LOOP'
        self.prev_center_error = 0.0
        self.last_stable_angular = 0.90 * self.last_stable_angular
        return 0.0

    def begin_turn(self, turn_dir: TurnDir):
        if not self.gyro_is_fresh() or self.current_yaw_deg is None:
            self.get_logger().warn('Cannot begin turn: gyro not fresh')
            return False

        if turn_dir == TurnDir.LEFT:
            delta = 90.0
        elif turn_dir == TurnDir.RIGHT:
            delta = -90.0
        elif turn_dir == TurnDir.BACK:
            delta = 180.0
        else:
            return False

        self.turn_direction = turn_dir
        self.turn_target_yaw_deg = self.wrap_angle_deg(self.current_yaw_deg + delta)
        self.turn_settle_counter = 0
        self.nav_state = NavState.TURNING
        self.target_yaw_deg = self.turn_target_yaw_deg
        self.prev_heading_error = 0.0
        self.prev_angular_cmd = 0.0

        self.get_logger().info(
            f'Begin turn {turn_dir.value} | current={self.current_yaw_deg:.1f} '
            f'target={self.turn_target_yaw_deg:.1f}'
        )
        return True

    def execute_turn(self):
        twist = Twist()

        if not self.gyro_is_fresh() or self.current_yaw_deg is None or self.turn_target_yaw_deg is None:
            self.stop_cmd()
            return

        err = self.angle_error_deg(self.turn_target_yaw_deg, self.current_yaw_deg)
        abs_err = abs(err)

        if abs_err <= self.turn_tolerance_deg:
            self.turn_settle_counter += 1
        else:
            self.turn_settle_counter = 0

        if self.turn_settle_counter >= self.turn_settle_cycles:
            self.stop_cmd()

            next_state = NavState.RETURN_TO_JUNCTION if self.return_after_back_turn else NavState.FOLLOW
            self.return_after_back_turn = False

            self.nav_state = next_state
            self.turn_direction = TurnDir.NONE
            self.turn_target_yaw_deg = None
            self.prev_angular_cmd = 0.0
            self.last_junction_time = self.now_sec()

            if next_state == NavState.FOLLOW:
                self.post_turn_forward_cycles = max(
                    1, int(self.post_turn_forward_time_sec * self.control_rate_hz)
                )
            else:
                self.post_turn_forward_cycles = 0

            self.capture_heading_reference('turn_complete')
            self.reset_ff_trackers()
            return

        ang_mag = self.turn_angular_speed
        if abs_err < self.turn_slowdown_error_deg:
            ang_mag = self.turn_slow_angular_speed

        twist.linear.x = 0.0
        twist.angular.z = ang_mag if err > 0.0 else -ang_mag
        self.cmd_pub.publish(twist)

    def begin_left_entry(self):
        self.left_entry_start_distance_m = self.distance_since_reset_m
        self.left_entry_target_distance_m = self.distance_since_reset_m + self.left_entry_distance_m
        self.nav_state = NavState.ENTER_LEFT_JUNCTION
        self.debug_print(
            f'[LEFT_ENTRY_START] start={self.left_entry_start_distance_m:.3f} '
            f'target={self.left_entry_target_distance_m:.3f}'
        )

    def execute_left_entry(self):
        if self.front_valid and self.front_range is not None:
            if self.front_range <= (self.front_stop_distance + self.front_stop_hysteresis):
                self.stop_cmd()
                self.begin_turn(TurnDir.LEFT)
                return

        if self.distance_since_reset_m >= self.left_entry_target_distance_m:
            self.stop_cmd()
            self.begin_turn(TurnDir.LEFT)
            return

        twist = Twist()
        twist.linear.x = self.left_entry_speed
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def choose_turn_left_priority(self, front_open: bool, left_open: bool, right_open: bool):
        if left_open:
            return TurnDir.LEFT
        if front_open:
            return TurnDir.NONE
        if right_open:
            return TurnDir.RIGHT
        return TurnDir.BACK

    def choose_turn_when_front_blocked(self):
        left_open = self.left_open_for_turn()
        right_open = self.right_open_for_turn()
        front_open = self.front_open_for_turn()

        decision = self.choose_turn_left_priority(front_open, left_open, right_open)

        if decision == TurnDir.BACK:
            return TurnDir.BACK, left_open, right_open

        self.count_block_or_dead_end_event(left_open, right_open)
        return decision, left_open, right_open

    def refresh_for_decision_step(self):
        self.stop_cmd()
        if self.both_ready_for_decision():
            self.nav_state = NavState.STOP_AND_DECIDE

    def start_dead_end_return(self):
        target = max(0.0, self.distance_since_reset_m)

        if not self.begin_turn(TurnDir.BACK):
            return False

        if self.count_dead_ends:
            self.dead_end_count += 1
            self.publish_counts()

        self.return_distance_target_m = target
        self.return_after_back_turn = True
        self.pending_block_event = False
        self.reset_distance_segment()
        return True

    def follow_motion(self):
        twist = Twist()

        wall_term = self.compute_wall_term()
        ff_term = self.compute_feedforward_term()
        self.maybe_update_target_heading(self.mode)
        heading_term = self.compute_heading_term(self.mode)

        if self.ff_pulse_active():
            heading_term *= self.ff_heading_suppress_gain

        wall_weight = 0.0
        if self.mode in ['BOTH_WALLS', 'LEFT_WALL', 'RIGHT_WALL']:
            wall_weight = self.wall_weight_both
            if self.front_valid and self.front_range is not None and self.front_range < self.near_front_fusion_distance:
                wall_weight = self.wall_weight_near_front

        raw_ang = wall_weight * wall_term + heading_term + ff_term

        self.debug_periodic(
            f'[FOLLOW_CTRL] mode={self.mode} '
            f'L={self.fmt(self.left_range)} Lraw={self.fmt(self.last_left_raw)} lockL={self.left_edge_locked} '
            f'R={self.fmt(self.right_range)} Rraw={self.fmt(self.last_right_raw)} lockR={self.right_edge_locked} '
            f'wall={wall_term:.3f} head={heading_term:.3f} ff={ff_term:.3f} raw={raw_ang:.3f} '
            f'yaw={self.fmt(self.current_yaw_deg)} target={self.fmt(self.target_yaw_deg)}'
        )

        raw_ang = self.clamp(raw_ang, -self.max_angular, self.max_angular)
        raw_ang = self.apply_deadband(raw_ang, self.angular_deadband)

        ang = self.slew_limit(raw_ang, self.prev_angular_cmd, self.angular_slew_per_cycle)
        ang = self.clamp(ang, -self.max_angular, self.max_angular)
        self.prev_angular_cmd = ang

        speed = self.compute_forward_speed(self.mode)

        if self.post_turn_forward_cycles > 0:
            self.post_turn_forward_cycles -= 1
            speed = min(speed, 0.06)

        if speed <= 0.0:
            if not self.stopped:
                self.stopped = True
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            return True

        self.stopped = False

        steer_scale = 1.0 - 0.30 * min(1.0, abs(ang) / max(1e-6, self.max_angular))
        speed *= steer_scale
        speed = max(self.min_forward_speed, speed)

        twist.linear.x = speed
        twist.angular.z = ang
        self.cmd_pub.publish(twist)
        return False

    def recent_junction_block(self):
        if self.last_junction_time is None:
            return False
        return (self.now_sec() - self.last_junction_time) < self.junction_cooldown_sec

    def control_loop(self):
        self.debug_cycle_counter += 1
        self.update_cell_count()

        if not (self.left_valid or self.front_valid or self.right_valid or self.gyro_is_fresh()):
            self.stop_cmd()
            return

        if self.nav_state == NavState.TURNING:
            self.execute_turn()
            return

        if self.nav_state == NavState.ENTER_LEFT_JUNCTION:
            self.execute_left_entry()
            return

        if self.nav_state == NavState.RETURN_TO_JUNCTION:
            stopped = self.follow_motion()

            if self.distance_since_reset_m >= max(0.0, self.return_distance_target_m - self.return_stop_margin_m):
                self.stop_cmd()
                target_heading = self.choose_return_target_heading()

                if target_heading is None:
                    self.nav_state = NavState.FOLLOW
                    self.reset_distance_segment()
                    self.capture_heading_reference('return_no_memory')
                    return

                self.last_junction_exit_heading_deg = target_heading
                turn_dir = self.heading_to_turn_dir(target_heading)

                self.reset_distance_segment()

                if turn_dir == TurnDir.NONE:
                    self.nav_state = NavState.FOLLOW
                    self.capture_heading_reference('return_straight')
                else:
                    self.begin_turn(turn_dir)
                return

            if stopped:
                self.stop_cmd()
            return

        if self.nav_state == NavState.REFRESH_FOR_DECISION:
            self.refresh_for_decision_step()
            return

        if self.nav_state == NavState.STOP_AND_DECIDE:
            self.stop_cmd()
            decision, left_open, right_open = self.choose_turn_when_front_blocked()

            if decision == TurnDir.BACK:
                self.start_dead_end_return()
            elif decision == TurnDir.NONE:
                self.nav_state = NavState.FOLLOW
                self.capture_heading_reference('decision_go_straight')
                self.last_junction_time = self.now_sec()
            elif decision == TurnDir.LEFT:
                self.remember_junction(False, left_open, right_open, TurnDir.LEFT)
                self.left_edge_locked = False
                self.right_edge_locked = False
                self.left_reacquire_counter = 0
                self.right_reacquire_counter = 0
                self.begin_left_entry()
            else:
                self.remember_junction(False, left_open, right_open, decision)
                self.left_edge_locked = False
                self.right_edge_locked = False
                self.left_reacquire_counter = 0
                self.right_reacquire_counter = 0
                self.begin_turn(decision)
            return

        self.update_wall_presence()

        left_open = self.left_open_for_turn()
        right_open = self.right_open_for_turn()
        front_open = self.front_open_for_turn()

        if not self.recent_junction_block():
            decision = self.choose_turn_left_priority(front_open, left_open, right_open)

            if decision == TurnDir.LEFT:
                if self.count_pass_junctions and self.can_count_new_event():
                    self.count_pass_junction_event(left_open, right_open)

                self.remember_junction(front_open, left_open, right_open, TurnDir.LEFT)

                self.left_edge_locked = False
                self.right_edge_locked = False
                self.left_reacquire_counter = 0
                self.right_reacquire_counter = 0

                self.last_junction_time = self.now_sec()
                self.begin_left_entry()
                return

            if decision == TurnDir.RIGHT and not front_open:
                if self.count_blocked_junctions and self.can_count_new_event():
                    self.junction_count += 1
                    self.blocked_junction_count += 1
                    self.publish_counts()
                    self.reset_distance_segment()

                self.remember_junction(front_open, left_open, right_open, TurnDir.RIGHT)

                self.left_edge_locked = False
                self.right_edge_locked = False
                self.left_reacquire_counter = 0
                self.right_reacquire_counter = 0

                self.begin_turn(TurnDir.RIGHT)
                self.last_junction_time = self.now_sec()
                return

            if decision == TurnDir.BACK and not front_open:
                stopped = self.follow_motion()
                if stopped:
                    self.begin_refresh_for_decision()
                return

            if front_open and (left_open or right_open):
                self.last_junction_time = self.now_sec()

        stopped = self.follow_motion()

        if stopped and not self.recent_junction_block():
            self.begin_refresh_for_decision()
            return

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


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