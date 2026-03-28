#!/usr/bin/env python3

import json
import math
from enum import Enum

import gpiod
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Int32, Empty, String


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

        # Junction distance / counting topics
        self.declare_parameter('distance_since_reset_topic', '/distance_since_reset')
        self.declare_parameter('reset_distance_topic', '/reset_distance')
        self.declare_parameter('junction_count_topic', '/junction_count')
        self.declare_parameter('dead_end_count_topic', '/dead_end_count')

        # Cell counting topic
        self.declare_parameter('cell_count_topic', '/cell_count')

        # =========================
        # Rates
        # =========================
        self.declare_parameter('control_rate_hz', 20.0)

        # =========================
        # Speeds
        # =========================
        self.declare_parameter('max_forward_speed', 0.14)
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
        self.declare_parameter('turn_open_distance', 0.34)
        self.declare_parameter('min_valid_range', 0.04)
        self.declare_parameter('max_valid_range', 16.0)
        self.declare_parameter('wall_loss_confirm_cycles', 5)

        # =========================
        # Wall control gains
        # =========================
        self.declare_parameter('center_kp', 2.2)
        self.declare_parameter('center_kd', 1.5)

        # =========================
        # Gyro heading-hold gains
        # =========================
        self.declare_parameter('heading_kp', 0.030)
        self.declare_parameter('heading_kd', 0.010)
        self.declare_parameter('heading_weight_both', 0.20)
        self.declare_parameter('heading_weight_missing', 1.00)
        self.declare_parameter('heading_capture_alpha', 0.12)
        self.declare_parameter('heading_error_limit_deg', 25.0)
        self.declare_parameter('gyro_valid_timeout_sec', 0.50)

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
        self.declare_parameter('angular_deadband', 0.03)
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
        self.declare_parameter('ff_enabled', True)
        self.declare_parameter('ff_trigger_delta', 0.005)
        self.declare_parameter('ff_trigger_cycles', 2)
        self.declare_parameter('ff_turn_mag', 0.35)
        self.declare_parameter('ff_hold_cycles', 6)
        self.declare_parameter('ff_cooldown_cycles', 5)
        self.declare_parameter('ff_front_min_distance', 0.1)
        self.declare_parameter('ff_edge_lock_block', True)
        self.declare_parameter('ff_heading_suppress_gain', 0.4)

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
        # AprilTag + LED
        # =========================
        self.declare_parameter('apriltag_topic', '/apriltag/decoded')
        self.declare_parameter('led_chip_name', 'gpiochip4')
        self.declare_parameter('led_pin', 26)
        self.declare_parameter('blink_half_period_sec', 0.3)

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

        self.center_kp = float(self.get_parameter('center_kp').value)
        self.center_kd = float(self.get_parameter('center_kd').value)

        self.heading_kp = float(self.get_parameter('heading_kp').value)
        self.heading_kd = float(self.get_parameter('heading_kd').value)
        self.heading_weight_both = float(self.get_parameter('heading_weight_both').value)
        self.heading_weight_missing = float(self.get_parameter('heading_weight_missing').value)
        self.heading_capture_alpha = float(self.get_parameter('heading_capture_alpha').value)
        self.heading_error_limit_deg = float(self.get_parameter('heading_error_limit_deg').value)
        self.gyro_valid_timeout_sec = float(self.get_parameter('gyro_valid_timeout_sec').value)

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

        self.apriltag_topic = self.get_parameter('apriltag_topic').value
        self.led_chip_name = str(self.get_parameter('led_chip_name').value)
        self.led_pin = int(self.get_parameter('led_pin').value)
        self.blink_half_period_sec = float(self.get_parameter('blink_half_period_sec').value)

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

        self.prev_center_error = 0.0
        self.prev_heading_error = 0.0
        self.prev_angular_cmd = 0.0
        self.last_stable_angular = 0.0

        self.corridor_width_est = None
        self.mode = 'INIT'
        self.stopped = False

        # Gyro/yaw state
        self.current_yaw_deg = None
        self.last_yaw_deg = None
        self.target_yaw_deg = None
        self.last_gyro_msg_time = None

        # High-level state
        self.nav_state = NavState.FOLLOW
        self.turn_direction = TurnDir.NONE
        self.turn_target_yaw_deg = None
        self.turn_settle_counter = 0
        self.post_turn_forward_cycles = 0
        self.last_junction_time = None

        # Edge lock state for side sensors
        self.left_edge_locked = False
        self.right_edge_locked = False

        # Refresh-for-decision state
        self.left_decision_range = None
        self.right_decision_range = None
        self.left_decision_samples = 0
        self.right_decision_samples = 0

        # Feed-forward tracking
        self.prev_left_ff_range = None
        self.prev_right_ff_range = None
        self.left_toward_wall_count = 0
        self.right_toward_wall_count = 0
        self.ff_active_dir = TurnDir.NONE
        self.ff_hold_counter = 0
        self.ff_cooldown_counter = 0

        # Debug state
        self.debug_cycle_counter = 0
        self.last_left_raw = None
        self.last_right_raw = None
        self.last_front_raw = None

        # Junction counting state
        self.distance_since_reset_m = 0.0
        self.junction_count = 0
        self.dead_end_count = 0
        self.pass_junction_count = 0
        self.blocked_junction_count = 0

        self.pending_block_event = False
        self.pending_block_event_distance = 0.0

        # Cell counting state
        self.cell_count = 0
        self.segment_cells_counted = 0

        # AprilTag detection state
        self.seen_tag_ids: set = set()
        self._tag_blink_queued = False

        # LED blink state
        self._led_line = None
        self._blink_count = 0
        self._blink_timer = None

        # ROS interfaces
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

        self._init_led()
        self.apriltag_sub = self.create_subscription(
            String, self.apriltag_topic, self._apriltag_cb, 10
        )

        self.timer = self.create_timer(1.0 / self.control_rate_hz, self.control_loop)

        self.get_logger().info('Task1 maze node started')
        self.get_logger().info(
            f'left={self.left_range_topic}, front={self.front_range_topic}, '
            f'right={self.right_range_topic}, gyro={self.gyro_angle_topic}'
        )
        self.get_logger().info(
            f'junction counting enabled | distance_topic={self.distance_since_reset_topic} '
            f'| min_spacing={self.junction_min_distance_m:.3f} m'
        )
        self.get_logger().info(
            f'cell counting enabled | cell_length_m={self.cell_length_m:.3f} '
            f'| cell_topic={self.cell_count_topic}'
        )

    # ============================================================
    # Helpers
    # ============================================================
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

    def filtered_update(self, old_val, new_val, alpha, max_jump):
        if old_val is None:
            return new_val
        if abs(new_val - old_val) > max_jump:
            return old_val
        return alpha * new_val + (1.0 - alpha) * old_val

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

    def front_blocked(self):
        return (
            self.front_valid and
            self.front_range is not None and
            self.front_range <= (self.front_stop_distance + self.front_stop_hysteresis)
        )

    def recent_junction_block(self):
        if self.last_junction_time is None:
            return False
        return (self.now_sec() - self.last_junction_time) < self.junction_cooldown_sec

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

            self.get_logger().info(
                f'[CELL_COUNT] total_cells={self.cell_count} '
                f'segment_cells={self.segment_cells_counted} '
                f'distance={self.distance_since_reset_m:.3f}'
            )

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
            self.debug_print(
                f'[JUNCTION_PASS_SKIP] distance={self.distance_since_reset_m:.3f} < '
                f'{self.junction_min_distance_m:.3f}'
            )
            return

        self.junction_count += 1
        self.pass_junction_count += 1
        self.publish_counts()

        side_txt = f'L={left_open}, R={right_open}'
        self.get_logger().info(
            f'[JUNCTION_PASS_COUNT] total={self.junction_count} '
            f'pass={self.pass_junction_count} dead_ends={self.dead_end_count} '
            f'distance={self.distance_since_reset_m:.3f} {side_txt}'
        )
        self.reset_distance_segment()

    def count_block_or_dead_end_event(self, left_open: bool, right_open: bool):
        if not self.pending_block_event:
            return

        if not self.can_count_new_event():
            self.debug_print(
                f'[JUNCTION_BLOCK_SKIP] distance={self.distance_since_reset_m:.3f} < '
                f'{self.junction_min_distance_m:.3f}'
            )
            self.pending_block_event = False
            return

        if left_open or right_open:
            if self.count_blocked_junctions:
                self.junction_count += 1
                self.blocked_junction_count += 1
                self.publish_counts()

                self.get_logger().info(
                    f'[JUNCTION_BLOCK_COUNT] total={self.junction_count} '
                    f'blocked={self.blocked_junction_count} dead_ends={self.dead_end_count} '
                    f'distance={self.distance_since_reset_m:.3f} '
                    f'L={left_open}, R={right_open}'
                )
        else:
            if self.count_dead_ends:
                self.dead_end_count += 1
                self.publish_counts()

                self.get_logger().info(
                    f'[DEAD_END_COUNT] total_junctions={self.junction_count} '
                    f'dead_ends={self.dead_end_count} distance={self.distance_since_reset_m:.3f}'
                )

        self.reset_distance_segment()
        self.pending_block_event = False

    def begin_refresh_for_decision(self):
        self.debug_print(
            f'[BEGIN_REFRESH] front={self.fmt(self.front_range)} '
            f'oldL={self.fmt(self.left_range)} oldR={self.fmt(self.right_range)} '
            f'lockL={self.left_edge_locked} lockR={self.right_edge_locked}'
        )

        # mark blocked event candidate here, but count only after fresh side check
        self.pending_block_event = True
        self.pending_block_event_distance = self.distance_since_reset_m

        # Release both locks
        self.left_edge_locked = False
        self.right_edge_locked = False

        # Throw away old side values so decision cannot use stale readings
        self.left_range = None
        self.right_range = None
        self.left_valid = False
        self.right_valid = False

        # Clear wall state memory for decision phase
        self.left_wall_present = False
        self.right_wall_present = False
        self.left_loss_counter = 0
        self.right_loss_counter = 0

        # Reset fresh decision accumulators
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

    # ============================================================
    # Wall presence
    # ============================================================
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

    # ============================================================
    # Sensor callbacks
    # ============================================================
    def left_cb(self, msg: Range):
        r = float(msg.range)
        self.last_left_raw = r

        if not self.is_valid_measurement(r):
            self.debug_print(f'[LEFT_CB] invalid raw={self.fmt(r)}')
            return

        if self.nav_state == NavState.REFRESH_FOR_DECISION:
            old_decision = self.left_decision_range
            self.left_decision_range = self.ema_update(
                self.left_decision_range, r, self.decision_filter_alpha
            )
            self.left_decision_samples += 1

            self.left_range = self.left_decision_range
            self.left_valid = True

            self.debug_print(
                f'[LEFT_REFRESH] raw={self.fmt(r)} old_dec={self.fmt(old_decision)} '
                f'new_dec={self.fmt(self.left_decision_range)} samples={self.left_decision_samples}'
            )
            return

        if self.left_range is None:
            self.left_range = r
            self.debug_print(f'[LEFT_INIT] raw={self.fmt(r)} set={self.fmt(self.left_range)}')
        else:
            if self.left_edge_locked:
                self.debug_print(
                    f'[LEFT_LOCKED] raw={self.fmt(r)} hold={self.fmt(self.left_range)}'
                )
            else:
                diff = abs(r - self.left_range)
                if diff > self.max_side_jump:
                    self.left_edge_locked = True
                    self.debug_print(
                        f'[LEFT_EDGE_LOCK] raw={self.fmt(r)} cur={self.fmt(self.left_range)} '
                        f'diff={diff:.3f} max_jump={self.max_side_jump:.3f}'
                    )
                else:
                    old_val = self.left_range
                    self.left_range = self.ema_update(
                        self.left_range, r, self.side_filter_alpha
                    )
                    self.debug_print(
                        f'[LEFT_UPDATE] raw={self.fmt(r)} old={self.fmt(old_val)} '
                        f'new={self.fmt(self.left_range)} diff={diff:.3f}'
                    )

        self.left_valid = True

    def front_cb(self, msg: Range):
        r = float(msg.range)
        self.last_front_raw = r

        if self.is_valid_measurement(r):
            old_val = self.front_range
            self.front_range = self.filtered_update(
                self.front_range, r, self.front_filter_alpha, self.max_front_jump
            )
            self.front_valid = True

            self.debug_print(
                f'[FRONT_CB] raw={self.fmt(r)} old={self.fmt(old_val)} new={self.fmt(self.front_range)}'
            )
        else:
            self.debug_print(f'[FRONT_CB] invalid raw={self.fmt(r)}')

    def right_cb(self, msg: Range):
        r = float(msg.range)
        self.last_right_raw = r

        if not self.is_valid_measurement(r):
            self.debug_print(f'[RIGHT_CB] invalid raw={self.fmt(r)}')
            return

        if self.nav_state == NavState.REFRESH_FOR_DECISION:
            old_decision = self.right_decision_range
            self.right_decision_range = self.ema_update(
                self.right_decision_range, r, self.decision_filter_alpha
            )
            self.right_decision_samples += 1

            self.right_range = self.right_decision_range
            self.right_valid = True

            self.debug_print(
                f'[RIGHT_REFRESH] raw={self.fmt(r)} old_dec={self.fmt(old_decision)} '
                f'new_dec={self.fmt(self.right_decision_range)} samples={self.right_decision_samples}'
            )
            return

        if self.right_range is None:
            self.right_range = r
            self.debug_print(f'[RIGHT_INIT] raw={self.fmt(r)} set={self.fmt(self.right_range)}')
        else:
            if self.right_edge_locked:
                self.debug_print(
                    f'[RIGHT_LOCKED] raw={self.fmt(r)} hold={self.fmt(self.right_range)}'
                )
            else:
                diff = abs(r - self.right_range)
                if diff > self.max_side_jump:
                    self.right_edge_locked = True
                    self.debug_print(
                        f'[RIGHT_EDGE_LOCK] raw={self.fmt(r)} cur={self.fmt(self.right_range)} '
                        f'diff={diff:.3f} max_jump={self.max_side_jump:.3f}'
                    )
                else:
                    old_val = self.right_range
                    self.right_range = self.ema_update(
                        self.right_range, r, self.side_filter_alpha
                    )
                    self.debug_print(
                        f'[RIGHT_UPDATE] raw={self.fmt(r)} old={self.fmt(old_val)} '
                        f'new={self.fmt(self.right_range)} diff={diff:.3f}'
                    )

        self.right_valid = True

    def gyro_cb(self, msg: Float32):
        yaw_deg = float(msg.data)
        yaw_deg = self.wrap_angle_deg(yaw_deg)
        self.last_yaw_deg = self.current_yaw_deg
        self.current_yaw_deg = yaw_deg
        self.last_gyro_msg_time = self.get_clock().now()

        if self.target_yaw_deg is None:
            self.target_yaw_deg = yaw_deg

    # ============================================================
    # Speed control
    # ============================================================
    def compute_forward_speed(self, mode):
        if not self.front_valid or self.front_range is None:
            if mode == 'GYRO_ONLY':
                return self.blind_forward_speed
            if mode == 'TRANSITION':
                return self.transition_forward_speed
            return min(0.08, self.max_forward_speed)

        if self.front_range <= (self.front_stop_distance + self.front_stop_hysteresis):
            return 0.0

        if mode == 'GYRO_ONLY':
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

    # ============================================================
    # Heading-hold fusion
    # ============================================================
    def maybe_update_target_heading(self, mode):
        if not self.gyro_is_fresh() or self.current_yaw_deg is None:
            return

        if mode == 'BOTH_WALLS':
            if self.target_yaw_deg is None:
                self.target_yaw_deg = self.current_yaw_deg
            else:
                err = self.angle_error_deg(self.current_yaw_deg, self.target_yaw_deg)
                self.target_yaw_deg = self.wrap_angle_deg(
                    self.target_yaw_deg + self.heading_capture_alpha * err
                )

    def compute_heading_term(self, mode):
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

    # ============================================================
    # Feed-forward wall escape
    # ============================================================
    def reset_ff_trackers(self):
        self.left_toward_wall_count = 0
        self.right_toward_wall_count = 0
        self.prev_left_ff_range = self.left_range
        self.prev_right_ff_range = self.right_range

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

    def start_ff_pulse(self, direction: TurnDir, reason: str):
        self.ff_active_dir = direction
        self.ff_hold_counter = self.ff_hold_cycles
        self.ff_cooldown_counter = self.ff_cooldown_cycles
        self.debug_print(f'[FF_START] dir={direction.value} reason={reason}')

    def ff_pulse_active(self):
        return self.ff_hold_counter > 0

    def compute_feedforward_term(self):
        if self.ff_hold_counter > 0:
            self.ff_hold_counter -= 1

            if self.ff_active_dir == TurnDir.RIGHT:
                ang = -self.ff_turn_mag
            elif self.ff_active_dir == TurnDir.LEFT:
                ang = self.ff_turn_mag
            else:
                ang = 0.0

            if self.ff_hold_counter == 0:
                self.debug_print(f'[FF_END] dir={self.ff_active_dir.value}')
                self.ff_active_dir = TurnDir.NONE

            return ang

        if self.ff_cooldown_counter > 0:
            self.ff_cooldown_counter -= 1

        left = self.left_wall_present and self.left_valid and self.left_range is not None
        right = self.right_wall_present and self.right_valid and self.right_range is not None

        if self.ff_edge_lock_block:
            if self.left_edge_locked:
                left = False
            if self.right_edge_locked:
                right = False

        if not self.can_use_ff():
            self.reset_ff_trackers()
            return 0.0

        if left and not right:
            prev = self.prev_left_ff_range
            if prev is not None and self.left_range < (prev - self.ff_trigger_delta):
                self.left_toward_wall_count += 1
            else:
                self.left_toward_wall_count = max(0, self.left_toward_wall_count - 1)

            self.prev_left_ff_range = self.left_range
            self.prev_right_ff_range = self.right_range
            self.right_toward_wall_count = 0

            self.debug_periodic(
                f'[FF_MONITOR_LEFT] left={self.fmt(self.left_range)} '
                f'prev={self.fmt(prev)} count={self.left_toward_wall_count}'
            )

            if self.left_toward_wall_count >= self.ff_trigger_cycles:
                self.left_toward_wall_count = 0
                self.start_ff_pulse(TurnDir.RIGHT, 'pushing_toward_left_wall')
                return -self.ff_turn_mag

            return 0.0

        if right and not left:
            prev = self.prev_right_ff_range
            if prev is not None and self.right_range < (prev - self.ff_trigger_delta):
                self.right_toward_wall_count += 1
            else:
                self.right_toward_wall_count = max(0, self.right_toward_wall_count - 1)

            self.prev_right_ff_range = self.right_range
            self.prev_left_ff_range = self.left_range
            self.left_toward_wall_count = 0

            self.debug_periodic(
                f'[FF_MONITOR_RIGHT] right={self.fmt(self.right_range)} '
                f'prev={self.fmt(prev)} count={self.right_toward_wall_count}'
            )

            if self.right_toward_wall_count >= self.ff_trigger_cycles:
                self.right_toward_wall_count = 0
                self.start_ff_pulse(TurnDir.LEFT, 'pushing_toward_right_wall')
                return self.ff_turn_mag

            return 0.0

        self.reset_ff_trackers()
        return 0.0

    # ============================================================
    # Wall steering
    # ============================================================
    def compute_wall_term(self):
        left = self.left_wall_present and (not self.left_edge_locked)
        right = self.right_wall_present and (not self.right_edge_locked)

        if left and right and self.left_range is not None and self.right_range is not None:
            measured_width = self.left_range + self.right_range
            if self.corridor_width_est is None:
                self.corridor_width_est = measured_width
            else:
                a = self.corridor_width_alpha
                self.corridor_width_est = a * measured_width + (1.0 - a) * self.corridor_width_est

        if left and right:
            self.mode = 'BOTH_WALLS'
            error = self.left_range - self.right_range
            derivative = (error - self.prev_center_error) * self.control_rate_hz
            self.prev_center_error = error

            ang = self.center_kp * error + self.center_kd * derivative
            self.last_stable_angular = 0.85 * self.last_stable_angular + 0.15 * ang
            return ang

        self.mode = 'GYRO_ONLY'
        self.prev_center_error = 0.0
        self.last_stable_angular = 0.90 * self.last_stable_angular
        return 0.0

    # ============================================================
    # Turning
    # ============================================================
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
            self.nav_state = NavState.FOLLOW
            self.turn_direction = TurnDir.NONE
            self.turn_target_yaw_deg = None
            self.prev_angular_cmd = 0.0
            self.last_junction_time = self.now_sec()
            self.post_turn_forward_cycles = max(
                1, int(self.post_turn_forward_time_sec * self.control_rate_hz)
            )
            self.get_logger().info('Turn complete')
            return

        ang_mag = self.turn_angular_speed
        if abs_err < self.turn_slowdown_error_deg:
            ang_mag = self.turn_slow_angular_speed

        twist.linear.x = 0.0
        twist.angular.z = ang_mag if err > 0.0 else -ang_mag
        self.cmd_pub.publish(twist)

    # ============================================================
    # Decision logic
    # ============================================================
    def choose_turn_when_front_blocked(self):
        left_open = self.left_valid and self.left_range is not None and self.left_range > self.turn_open_distance
        right_open = self.right_valid and self.right_range is not None and self.right_range > self.turn_open_distance

        self.count_block_or_dead_end_event(left_open, right_open)

        self.debug_print(
            f'[DECIDE] front={self.fmt(self.front_range)} '
            f'left_raw={self.fmt(self.last_left_raw)} left={self.fmt(self.left_range)} left_valid={self.left_valid} left_samples={self.left_decision_samples} left_open={left_open} '
            f'right_raw={self.fmt(self.last_right_raw)} right={self.fmt(self.right_range)} right_valid={self.right_valid} right_samples={self.right_decision_samples} right_open={right_open}'
        )

        if left_open and not right_open:
            self.get_logger().info(f'Front blocked, left open ({self.left_range:.3f}) -> LEFT')
            return TurnDir.LEFT

        if right_open and not left_open:
            self.get_logger().info(f'Front blocked, right open ({self.right_range:.3f}) -> RIGHT')
            return TurnDir.RIGHT

        if left_open and right_open:
            choice = TurnDir.LEFT if self.prefer_left_first else TurnDir.RIGHT
            self.get_logger().info(
                f'Front blocked, both open | L={self.left_range:.3f}, R={self.right_range:.3f} -> {choice.value}'
            )
            return choice

        left_txt = f'{self.left_range:.3f}' if self.left_range is not None else 'None'
        right_txt = f'{self.right_range:.3f}' if self.right_range is not None else 'None'
        self.get_logger().info(f'Front blocked, no side open | L={left_txt}, R={right_txt} -> BACK')
        return TurnDir.BACK

    def refresh_for_decision_step(self):
        self.stop_cmd()

        self.debug_print(
            f'[REFRESH_STEP] front={self.fmt(self.front_range)} '
            f'Lraw={self.fmt(self.last_left_raw)} Ldec={self.fmt(self.left_decision_range)} Lsamples={self.left_decision_samples} Lready={self.left_ready_for_decision()} '
            f'Rraw={self.fmt(self.last_right_raw)} Rdec={self.fmt(self.right_decision_range)} Rsamples={self.right_decision_samples} Rready={self.right_ready_for_decision()}'
        )

        if self.both_ready_for_decision():
            self.debug_print('[REFRESH_STEP] both sides ready -> STOP_AND_DECIDE')
            self.nav_state = NavState.STOP_AND_DECIDE

    # ============================================================
    # Follow controller
    # ============================================================
    def follow_motion(self):
        twist = Twist()

        wall_term = self.compute_wall_term()
        ff_term = self.compute_feedforward_term()
        self.maybe_update_target_heading(self.mode)
        heading_term = self.compute_heading_term(self.mode)

        # Let FF pulse dominate more when active
        if self.ff_pulse_active():
            heading_term *= self.ff_heading_suppress_gain

        wall_weight = 0.0
        if self.mode == 'BOTH_WALLS':
            wall_weight = self.wall_weight_both
            if self.front_valid and self.front_range is not None and self.front_range < self.near_front_fusion_distance:
                wall_weight = self.wall_weight_near_front

        raw_ang = wall_weight * wall_term + heading_term + ff_term

        self.debug_periodic(
            f'[FOLLOW_CTRL] mode={self.mode} wall={wall_term:.3f} head={heading_term:.3f} '
            f'ff={ff_term:.3f} ff_active={self.ff_pulse_active()} raw={raw_ang:.3f}'
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
                dist = self.front_range if self.front_range is not None else -1.0
                self.get_logger().info(f'Stopped at front distance: {dist:.3f} m')
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

    # ============================================================
    # AprilTag detection + LED blink
    # ============================================================
    def _init_led(self):
        """Open the gpiod LED line. Logs a warning and continues if it fails."""
        try:
            chip = gpiod.Chip(self.led_chip_name)
            self._led_line = chip.get_line(self.led_pin)
            self._led_line.request(
                consumer='task1_led',
                type=gpiod.LINE_REQ_DIR_OUT,
                default_vals=[0]
            )
            self.get_logger().info(
                f'Red LED initialized on {self.led_chip_name} pin {self.led_pin}'
            )
        except Exception as exc:
            self._led_line = None
            self.get_logger().warn(
                f'LED init failed ({self.led_chip_name} pin {self.led_pin}): {exc}. '
                'Tag detection will log only — no physical blink.'
            )

    def _set_led(self, on: bool):
        if self._led_line is None:
            return
        try:
            self._led_line.set_value(1 if on else 0)
        except Exception as exc:
            self.get_logger().warn(f'LED set_value error: {exc}')

    def _start_tag_blink(self):
        """Stop the robot and blink the red LED twice to signal tag detection."""
        if self._blink_timer is not None:
            self._blink_timer.cancel()
            self._blink_timer = None

        self._blink_count = 0
        self.nav_state = NavState.TAG_BLINK
        self._set_led(True)
        self._blink_timer = self.create_timer(self.blink_half_period_sec, self._blink_step)
        self.get_logger().info('AprilTag detected — stopped, blinking red LED ×2.')

    def _blink_step(self):
        """Timer callback: drives ON→OFF→ON→OFF sequence then resumes navigation."""
        self._blink_count += 1
        led_on = (self._blink_count % 2 == 0)
        self._set_led(led_on)

        if self._blink_count >= 3:
            self._set_led(False)
            self._blink_timer.cancel()
            self._blink_timer = None
            self.nav_state = NavState.FOLLOW
            self.get_logger().info('Blink complete — resuming navigation.')

    def _apriltag_cb(self, msg: String):
        """Called whenever apriltag_decoder publishes a new detection."""
        try:
            payload = json.loads(msg.data)
        except Exception:
            return

        incoming_ids = set(payload.get('tag_ids', []))
        new_ids = incoming_ids - self.seen_tag_ids
        if not new_ids:
            return

        self.seen_tag_ids |= new_ids
        decoded = payload.get('decoded_by_order', {})
        self.get_logger().info(
            f'[APRILTAG] New tag(s): {sorted(new_ids)} | '
            f'total seen={len(self.seen_tag_ids)}/8 | decoded={decoded}'
        )

        if self.nav_state == NavState.FOLLOW:
            self._start_tag_blink()
        else:
            self._tag_blink_queued = True

    # ============================================================
    # Control loop
    # ============================================================
    def control_loop(self):
        self.debug_cycle_counter += 1

        # update cell count continuously from straight-line distance
        self.update_cell_count()

        if not (self.left_valid or self.front_valid or self.right_valid or self.gyro_is_fresh()):
            self.stop_cmd()
            return

        if self.nav_state == NavState.TURNING:
            self.execute_turn()
            return

        if self.nav_state == NavState.REFRESH_FOR_DECISION:
            self.refresh_for_decision_step()
            return

        if self.nav_state == NavState.STOP_AND_DECIDE:
            self.stop_cmd()
            decision = self.choose_turn_when_front_blocked()
            self.begin_turn(decision)
            return

        if self.nav_state == NavState.TAG_BLINK:
            self.stop_cmd()
            return

        # FOLLOW
        if self._tag_blink_queued:
            self._tag_blink_queued = False
            self._start_tag_blink()
            return

        self.update_wall_presence()

        left_open = self.left_valid and self.left_range is not None and self.left_range > self.turn_open_distance
        right_open = self.right_valid and self.right_range is not None and self.right_range > self.turn_open_distance
        front_open = (
            self.front_valid and self.front_range is not None and
            self.front_range > (self.front_stop_distance + self.front_stop_hysteresis)
        )

        # Count pass-through junctions during follow
        if (
            front_open and
            (left_open or right_open) and
            not self.recent_junction_block()
        ):
            self.count_pass_junction_event(left_open, right_open)
            self.last_junction_time = self.now_sec()

        self.debug_periodic(
            f'[FOLLOW] front_raw={self.fmt(self.last_front_raw)} front={self.fmt(self.front_range)} '
            f'left_raw={self.fmt(self.last_left_raw)} left={self.fmt(self.left_range)} lockL={self.left_edge_locked} wallL={self.left_wall_present} '
            f'right_raw={self.fmt(self.last_right_raw)} right={self.fmt(self.right_range)} lockR={self.right_edge_locked} wallR={self.right_wall_present} '
            f'mode={self.mode} ffHold={self.ff_hold_counter} ffCooldown={self.ff_cooldown_counter} '
            f'dist={self.distance_since_reset_m:.3f} cells={self.cell_count} seg_cells={self.segment_cells_counted} '
            f'junc={self.junction_count} dead={self.dead_end_count}'
        )

        stopped = self.follow_motion()

        if stopped and not self.recent_junction_block():
            self.debug_print('[CONTROL] front stop detected -> begin refresh for decision')
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