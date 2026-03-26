#!/usr/bin/env python3

import math
from dataclasses import dataclass, field
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


class NavState(Enum):
    FOLLOW = 0
    STOP_AND_DECIDE = 1
    TURNING = 2
    FINISHED = 3


class TurnDir(Enum):
    LEFT = 'L'
    FRONT = 'F'
    RIGHT = 'R'
    BACK = 'B'
    NONE = 'N'


@dataclass
class JunctionFrame:
    # Directions at this junction that were not taken yet
    remaining_dirs: list = field(default_factory=list)


class Task1MazeDFSNode(Node):
    """
    Fixed-arena Task 1 traversal node.

    Strategy:
    1. Follow corridors using:
       - double-wall centering when both walls exist
       - gyro heading hold otherwise
    2. When a junction/dead-end is detected:
       - choose unexplored branch in priority LEFT -> FRONT -> RIGHT
       - save remaining branches in stack
    3. At dead-end:
       - turn BACK
       - return to previous junction
    4. When all junction branches are explored:
       - run configurable exit_plan
    """

    def __init__(self):
        super().__init__('task1_dfs')

        # =========================================================
        # Topics
        # =========================================================
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('left_range_topic', '/robocop/ds_left')
        self.declare_parameter('front_range_topic', '/robocop/ds_front')
        self.declare_parameter('right_range_topic', '/robocop/ds_right')
        self.declare_parameter('gyro_angle_topic', '/gyro_angle')

        # =========================================================
        # Rates
        # =========================================================
        self.declare_parameter('control_rate_hz', 20.0)

        # =========================================================
        # Speeds
        # =========================================================
        self.declare_parameter('max_forward_speed', 0.14)
        self.declare_parameter('transition_forward_speed', 0.08)
        self.declare_parameter('blind_forward_speed', 0.06)
        self.declare_parameter('min_forward_speed', 0.03)

        # =========================================================
        # Front stop / slowdown
        # =========================================================
        self.declare_parameter('front_stop_distance', 0.20)
        self.declare_parameter('front_slow_distance', 0.35)
        self.declare_parameter('front_hard_slow_distance', 0.28)
        self.declare_parameter('front_stop_hysteresis', 0.01)

        # =========================================================
        # Wall handling
        # =========================================================
        self.declare_parameter('wall_detect_distance', 0.20)
        self.declare_parameter('wall_clear_distance', 0.30)
        self.declare_parameter('turn_open_distance', 0.34)
        self.declare_parameter('min_valid_range', 0.04)
        self.declare_parameter('max_valid_range', 1.50)
        self.declare_parameter('wall_loss_confirm_cycles', 5)

        # =========================================================
        # Wall control gains
        # =========================================================
        self.declare_parameter('center_kp', 2.2)
        self.declare_parameter('center_kd', 1.5)

        # =========================================================
        # Gyro heading-hold gains
        # =========================================================
        self.declare_parameter('heading_kp', 0.030)
        self.declare_parameter('heading_kd', 0.010)
        self.declare_parameter('heading_weight_both', 0.20)
        self.declare_parameter('heading_weight_missing', 1.00)
        self.declare_parameter('heading_capture_alpha', 0.12)
        self.declare_parameter('heading_error_limit_deg', 25.0)
        self.declare_parameter('gyro_valid_timeout_sec', 0.50)

        # =========================================================
        # Fusion shaping
        # =========================================================
        self.declare_parameter('wall_weight_both', 1.00)
        self.declare_parameter('wall_weight_near_front', 0.35)
        self.declare_parameter('near_front_fusion_distance', 0.30)

        # =========================================================
        # Output shaping
        # =========================================================
        self.declare_parameter('max_angular', 0.8)
        self.declare_parameter('angular_deadband', 0.03)
        self.declare_parameter('angular_slew_per_cycle', 0.10)

        # =========================================================
        # Filtering / edge handling
        # =========================================================
        self.declare_parameter('side_filter_alpha', 0.22)
        self.declare_parameter('front_filter_alpha', 0.30)
        self.declare_parameter('max_side_jump', 0.14)
        self.declare_parameter('max_front_jump', 0.22)
        self.declare_parameter('corridor_width_alpha', 0.10)

        # =========================================================
        # Turn controller
        # =========================================================
        self.declare_parameter('turn_angular_speed', 0.45)
        self.declare_parameter('turn_slow_angular_speed', 0.22)
        self.declare_parameter('turn_slowdown_error_deg', 18.0)
        self.declare_parameter('turn_tolerance_deg', 3.0)
        self.declare_parameter('turn_settle_cycles', 4)
        self.declare_parameter('post_turn_forward_time_sec', 0.35)
        self.declare_parameter('junction_cooldown_sec', 0.80)

        # =========================================================
        # Junction / exploration behavior
        # =========================================================
        self.declare_parameter('decision_hold_time_sec', 0.18)
        self.declare_parameter('prefer_left_first', True)
        self.declare_parameter('stop_at_end', True)

        # Example exit plan after exploration is complete.
        # IMPORTANT:
        # Tune this after one physical run if needed.
        # Allowed tokens: L, F, R, B
        #
        # Example:
        # - []      -> stop when full exploration finishes
        # - ['R']   -> at final completion, take a right branch and continue
        # - ['R','F'] etc.
        #
        self.declare_parameter('exit_plan', [])

        # =========================================================
        # Read params
        # =========================================================
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.left_range_topic = self.get_parameter('left_range_topic').value
        self.front_range_topic = self.get_parameter('front_range_topic').value
        self.right_range_topic = self.get_parameter('right_range_topic').value
        self.gyro_angle_topic = self.get_parameter('gyro_angle_topic').value

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

        self.turn_angular_speed = float(self.get_parameter('turn_angular_speed').value)
        self.turn_slow_angular_speed = float(self.get_parameter('turn_slow_angular_speed').value)
        self.turn_slowdown_error_deg = float(self.get_parameter('turn_slowdown_error_deg').value)
        self.turn_tolerance_deg = float(self.get_parameter('turn_tolerance_deg').value)
        self.turn_settle_cycles = int(self.get_parameter('turn_settle_cycles').value)
        self.post_turn_forward_time_sec = float(self.get_parameter('post_turn_forward_time_sec').value)
        self.junction_cooldown_sec = float(self.get_parameter('junction_cooldown_sec').value)

        self.decision_hold_time_sec = float(self.get_parameter('decision_hold_time_sec').value)
        self.prefer_left_first = bool(self.get_parameter('prefer_left_first').value)
        self.stop_at_end = bool(self.get_parameter('stop_at_end').value)

        raw_exit_plan = list(self.get_parameter('exit_plan').value)
        self.exit_plan = [self.parse_turn_token(x) for x in raw_exit_plan if self.parse_turn_token(x) is not None]
        self.exit_plan_index = 0

        # =========================================================
        # State
        # =========================================================
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

        # Gyro / yaw state
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

        # DFS / traversal memory
        self.junction_stack = []
        self.backtracking = False
        self.exploration_done = False
        self.in_exit_mode = False
        self.at_root_started = False
        self.decision_wait_until = None

        # =========================================================
        # ROS interfaces
        # =========================================================
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

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.timer = self.create_timer(1.0 / self.control_rate_hz, self.control_loop)

        self.get_logger().info('Task1 DFS maze node started')
        self.get_logger().info(
            f'left={self.left_range_topic}, front={self.front_range_topic}, '
            f'right={self.right_range_topic}, gyro={self.gyro_angle_topic}'
        )
        self.get_logger().info(f'Configured exit_plan: {[x.value for x in self.exit_plan]}')

    # ============================================================
    # Helper utilities
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

    def left_open(self):
        return self.left_valid and self.left_range is not None and self.left_range > self.turn_open_distance

    def right_open(self):
        return self.right_valid and self.right_range is not None and self.right_range > self.turn_open_distance

    def recent_junction_block(self):
        if self.last_junction_time is None:
            return False
        return (self.now_sec() - self.last_junction_time) < self.junction_cooldown_sec

    def parse_turn_token(self, token):
        if token is None:
            return None
        s = str(token).strip().upper()
        if s == 'L':
            return TurnDir.LEFT
        if s == 'F':
            return TurnDir.FRONT
        if s == 'R':
            return TurnDir.RIGHT
        if s == 'B':
            return TurnDir.BACK
        return None

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
        if self.is_valid_measurement(r):
            self.left_range = self.filtered_update(
                self.left_range, r, self.side_filter_alpha, self.max_side_jump
            )
            self.left_valid = True

    def front_cb(self, msg: Range):
        r = float(msg.range)
        if self.is_valid_measurement(r):
            self.front_range = self.filtered_update(
                self.front_range, r, self.front_filter_alpha, self.max_front_jump
            )
            self.front_valid = True

    def right_cb(self, msg: Range):
        r = float(msg.range)
        if self.is_valid_measurement(r):
            self.right_range = self.filtered_update(
                self.right_range, r, self.side_filter_alpha, self.max_side_jump
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
    # Wall steering
    # ============================================================
    def compute_wall_term(self):
        left = self.left_wall_present
        right = self.right_wall_present

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
        if turn_dir == TurnDir.FRONT:
            # No actual rotation; just continue in current heading
            self.nav_state = NavState.FOLLOW
            self.turn_direction = TurnDir.NONE
            self.turn_target_yaw_deg = None
            self.turn_settle_counter = 0
            self.prev_heading_error = 0.0
            self.prev_angular_cmd = 0.0
            self.last_junction_time = self.now_sec()
            self.post_turn_forward_cycles = max(
                1, int(self.post_turn_forward_time_sec * self.control_rate_hz)
            )
            return True

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
    # Junction logic
    # ============================================================
    def available_forward_dirs(self):
        dirs = []
        if self.left_open():
            dirs.append(TurnDir.LEFT)
        if not self.front_blocked():
            dirs.append(TurnDir.FRONT)
        if self.right_open():
            dirs.append(TurnDir.RIGHT)
        return dirs

    def sort_dirs(self, dirs):
        if self.prefer_left_first:
            order = [TurnDir.LEFT, TurnDir.FRONT, TurnDir.RIGHT]
        else:
            order = [TurnDir.RIGHT, TurnDir.FRONT, TurnDir.LEFT]
        return [d for d in order if d in dirs]

    def is_decision_point(self):
        """
        Decision point when:
        - front blocked (dead-end / T / corner)
        - left side open
        - right side open

        This lets the robot branch at side openings instead of only waiting
        until a front wall appears.
        """
        if self.front_blocked():
            return True
        if self.left_open():
            return True
        if self.right_open():
            return True
        return False

    def choose_next_turn(self):
        """
        DFS with stack, assuming a tree-like maze (no loops).

        backtracking = False:
            choose first unexplored direction
            save remaining sibling branches in stack

        backtracking = True:
            we just came back from a child branch
            at next junction:
              - if parent has remaining branch -> take it
              - else keep BACKtracking
        """
        forward_dirs = self.sort_dirs(self.available_forward_dirs())

        # Root first entry
        if not self.at_root_started:
            self.at_root_started = True
            if len(forward_dirs) > 1:
                chosen = forward_dirs[0]
                remaining = forward_dirs[1:]
                self.junction_stack.append(JunctionFrame(remaining_dirs=remaining))
            elif len(forward_dirs) == 1:
                chosen = forward_dirs[0]
            else:
                chosen = TurnDir.BACK

            self.get_logger().info(
                f'ROOT choose={chosen.value} stack={[f.remaining_dirs for f in self.junction_stack]}'
            )
            return chosen

        # Exit mode after full exploration
        if self.in_exit_mode:
            if self.exit_plan_index < len(self.exit_plan):
                chosen = self.exit_plan[self.exit_plan_index]
                self.exit_plan_index += 1
                self.get_logger().info(f'EXIT plan choose={chosen.value}')
                return chosen

            self.exploration_done = True
            return TurnDir.NONE

        # Backtracking mode: the next junction reached is parent
        if self.backtracking:
            if self.junction_stack:
                top = self.junction_stack[-1]
                if top.remaining_dirs:
                    chosen = top.remaining_dirs.pop(0)
                    self.backtracking = False
                    self.get_logger().info(
                        f'RETURN parent -> choose sibling {chosen.value}, remaining={top.remaining_dirs}'
                    )
                    return chosen
                else:
                    self.junction_stack.pop()
                    if self.junction_stack:
                        self.backtracking = True
                        self.get_logger().info('RETURN parent exhausted -> BACK again')
                        return TurnDir.BACK
                    else:
                        # We are back at root and exploration is complete
                        self.get_logger().info('Exploration complete at root')
                        self.in_exit_mode = True
                        if self.exit_plan_index < len(self.exit_plan):
                            chosen = self.exit_plan[self.exit_plan_index]
                            self.exit_plan_index += 1
                            self.backtracking = False
                            self.get_logger().info(f'EXIT after full exploration -> {chosen.value}')
                            return chosen

                        return TurnDir.NONE
            else:
                # No stack means already at root
                self.get_logger().info('Backtracking finished with empty stack')
                self.in_exit_mode = True
                if self.exit_plan_index < len(self.exit_plan):
                    chosen = self.exit_plan[self.exit_plan_index]
                    self.exit_plan_index += 1
                    self.backtracking = False
                    return chosen
                return TurnDir.NONE

        # Normal exploring mode
        if len(forward_dirs) == 0:
            self.backtracking = True
            self.get_logger().info('Dead-end -> BACK')
            return TurnDir.BACK

        chosen = forward_dirs[0]
        remaining = forward_dirs[1:]

        if remaining:
            self.junction_stack.append(JunctionFrame(remaining_dirs=remaining))
            self.get_logger().info(
                f'New junction choose={chosen.value}, save remaining={remaining}, stack_depth={len(self.junction_stack)}'
            )
        else:
            self.get_logger().info(f'Corridor / single-choice choose={chosen.value}')

        return chosen

    # ============================================================
    # Follow controller
    # ============================================================
    def follow_motion(self):
        twist = Twist()

        wall_term = self.compute_wall_term()
        self.maybe_update_target_heading(self.mode)
        heading_term = self.compute_heading_term(self.mode)

        wall_weight = 0.0
        if self.mode == 'BOTH_WALLS':
            wall_weight = self.wall_weight_both
            if (
                self.front_valid and
                self.front_range is not None and
                self.front_range < self.near_front_fusion_distance
            ):
                wall_weight = self.wall_weight_near_front

        raw_ang = wall_weight * wall_term + heading_term
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
    # High-level control loop
    # ============================================================
    def control_loop(self):
        self.update_wall_presence()

        if not (self.left_valid or self.front_valid or self.right_valid or self.gyro_is_fresh()):
            self.stop_cmd()
            return

        if self.nav_state == NavState.FINISHED:
            self.stop_cmd()
            return

        if self.nav_state == NavState.TURNING:
            self.execute_turn()
            return

        if self.nav_state == NavState.STOP_AND_DECIDE:
            self.stop_cmd()

            if self.decision_wait_until is None:
                self.decision_wait_until = self.now_sec() + self.decision_hold_time_sec
                return

            if self.now_sec() < self.decision_wait_until:
                return

            self.decision_wait_until = None
            decision = self.choose_next_turn()

            if decision == TurnDir.NONE:
                self.get_logger().info('Traversal finished')
                if self.stop_at_end:
                    self.nav_state = NavState.FINISHED
                    self.stop_cmd()
                else:
                    self.nav_state = NavState.FOLLOW
                return

            self.begin_turn(decision)
            return

        # Normal following
        _ = self.follow_motion()

        # Junction detection by side opening or front block
        if not self.recent_junction_block() and self.is_decision_point():
            self.nav_state = NavState.STOP_AND_DECIDE
            return

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = Task1MazeDFSNode()

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