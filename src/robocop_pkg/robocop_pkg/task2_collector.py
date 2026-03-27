#!/usr/bin/env python3
"""
task2_collector.py — Task 2 main FSM for SLRC 2026

Behaviour sequence
==================
  1. SCAN_FOR_BOXES   — camera scan for red box straight ahead (startup check)
  2. GOTO_LINE        — drive forward until white line detected
  3. FOLLOW_LINE      — P-control on white-line centroid; monitor side ToF for
                        a *significant change* (delta from rolling baseline) →
                        turn toward that side
  4. TURN_TO_BOX      — gyro-based 90° turn toward detected side
  5. CENTER_APPROACH  — camera-centre red box; creep forward; reduce speed when
                        angular correction is large so the robot arrives
                        straight-on; stop when blob area OR front ToF threshold
  6. REQUEST_PICK     — send /pick_box action
  7. WAIT_PICK        — wait for arm result
  8. REVERSE          — drive backward to return to white-line corridor
  9. TURN_BACK        — gyro-based 90° turn back to original heading
  10. SCAN_AFTER_PICK — quick camera scan for another box in front
  11. RESUME_LINE     — drive forward to rediscover white line → FOLLOW_LINE
  12. TASK_DONE       — publish DONE

Key design decisions
====================
* ToF *change* detection (not absolute threshold):
    Keep a rolling deque of N valid readings as a baseline.
    Trigger when current < baseline - delta  for M consecutive frames.
    This adapts to any corridor width automatically.

* Straight-approach guarantee:
    While approaching, angular_vel = -Kp * horizontal_error.
    Forward speed is scaled down by the angular correction magnitude:
        fwd = approach_speed * (1 - clamp(|ang| / max_ang, 0, 0.8))
    The robot decelerates and corrects heading before advancing —
    arriving with near-zero lateral error when it stops.

* Post-pick reverse:
    The robot drives back toward the corridor (reverse_distance m)
    before turning, so it re-acquires the white line cleanly.
"""

import collections
import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Range
from std_msgs.msg import String, Float32

from robocop_pkg.line_detection_utils import build_white_mask
from robot_arm_interfaces.action import PickBox


class Task2Collector(Node):

    # ── State labels ────────────────────────────────────────────────────────
    ST_SCAN        = 'SCAN_FOR_BOXES'
    ST_GOTO_LINE   = 'GOTO_LINE'
    ST_FOLLOW      = 'FOLLOW_LINE'
    ST_TURN_TO_BOX = 'TURN_TO_BOX'
    ST_APPROACH    = 'CENTER_APPROACH'
    ST_REQ_PICK    = 'REQUEST_PICK'
    ST_WAIT_PICK   = 'WAIT_PICK'
    ST_REVERSE     = 'REVERSE'
    ST_TURN_BACK   = 'TURN_BACK'
    ST_SCAN_AFTER  = 'SCAN_AFTER_PICK'
    ST_RESUME      = 'RESUME_LINE'
    ST_DONE        = 'TASK_DONE'

    def __init__(self):
        super().__init__('task2_collector')

        # ── Topics ──────────────────────────────────────────────────────────
        self.declare_parameter('image_topic',        '/camera/image/image_color')
        self.declare_parameter('cmd_vel_topic',       '/cmd_vel')
        self.declare_parameter('left_range_topic',    '/robocop/ds_left')
        self.declare_parameter('right_range_topic',   '/robocop/ds_right')
        self.declare_parameter('front_range_topic',   '/robocop/ds_front')
        self.declare_parameter('gyro_angle_topic',    '/gyro_angle')
        self.declare_parameter('task2_status_topic',  '/task2/status')

        # ── Line following ───────────────────────────────────────────────────
        self.declare_parameter('linear_speed',      0.12)   # m/s following line
        self.declare_parameter('kp',                0.004)  # P-gain line centre error
        self.declare_parameter('max_angular',       1.2)    # rad/s clamp
        self.declare_parameter('search_linear',     0.04)   # when line lost
        self.declare_parameter('search_angular',    0.35)   # when line lost
        self.declare_parameter('roi_y_start',       0.60)   # image fraction for ROI
        self.declare_parameter('min_area',          5000)   # min white-blob area px²

        # White line HSV
        self.declare_parameter('h_low',   0)
        self.declare_parameter('s_low',   0)
        self.declare_parameter('v_low',   180)
        self.declare_parameter('h_high',  180)
        self.declare_parameter('s_high',  70)
        self.declare_parameter('v_high',  255)

        # ── Red box HSV ──────────────────────────────────────────────────────
        self.declare_parameter('red_h1_low',   0)
        self.declare_parameter('red_h1_high',  12)
        self.declare_parameter('red_h2_low',   165)
        self.declare_parameter('red_h2_high',  180)
        self.declare_parameter('red_s_low',    70)
        self.declare_parameter('red_v_low',    40)
        self.declare_parameter('red_min_area', 400)   # px² smallest valid blob

        # ── ToF change detection (key tunable parameters) ────────────────────
        self.declare_parameter('tof_delta_threshold', 0.12)  # m drop → box detected
        self.declare_parameter('tof_baseline_window', 25)    # rolling-window size (frames)
        self.declare_parameter('tof_confirm_frames',  4)     # consecutive triggers needed
        self.declare_parameter('tof_filter_alpha',    0.25)  # low-pass smoothing α

        # ── Box approach control ─────────────────────────────────────────────
        self.declare_parameter('approach_speed',         0.07)  # m/s creep forward
        self.declare_parameter('approach_kp_ang',        0.005) # angular gain (centering)
        self.declare_parameter('approach_max_ang',       0.8)   # rad/s clamp
        self.declare_parameter('approach_stop_area',     9000)  # px² → stop and pick
        self.declare_parameter('approach_stop_front_tof',0.22)  # m  → stop and pick
        self.declare_parameter('approach_timeout_sec',   8.0)   # abort if too slow
        self.declare_parameter('approach_lost_frames',   15)    # abort if box lost
        self.declare_parameter('front_hard_stop_dist',   0.12)  # m safety

        # ── Scan timeouts ────────────────────────────────────────────────────
        self.declare_parameter('scan_timeout_sec',       2.5)
        self.declare_parameter('scan_after_timeout_sec', 2.0)

        # ── Gyro turn ────────────────────────────────────────────────────────
        self.declare_parameter('turn_angle_deg',     90.0)
        self.declare_parameter('turn_tolerance_deg', 3.0)
        self.declare_parameter('turn_kp',            0.018)
        self.declare_parameter('turn_min_speed',     0.22)
        self.declare_parameter('turn_max_speed',     0.9)
        self.declare_parameter('turn_timeout_sec',   6.0)

        # ── Post-pick reverse ────────────────────────────────────────────────
        self.declare_parameter('reverse_speed',    0.12)  # m/s backward
        self.declare_parameter('reverse_distance', 0.55)  # m to travel back

        # ── Task control ─────────────────────────────────────────────────────
        self.declare_parameter('target_box_count',      6)
        self.declare_parameter('pick_action_name',      '/pick_box')
        self.declare_parameter('goto_line_speed',       0.10)  # m/s toward line
        self.declare_parameter('goto_line_timeout_sec', 10.0)
        self.declare_parameter('resume_forward_time',   2.0)   # s driving forward in RESUME
        self.declare_parameter('debug',                 False)

        # ── Read all parameters ──────────────────────────────────────────────
        def _p(name):
            return self.get_parameter(name).value

        self.image_topic       = _p('image_topic')
        self.cmd_vel_topic     = _p('cmd_vel_topic')
        self.left_range_topic  = _p('left_range_topic')
        self.right_range_topic = _p('right_range_topic')
        self.front_range_topic = _p('front_range_topic')
        self.gyro_angle_topic  = _p('gyro_angle_topic')
        self.status_topic      = _p('task2_status_topic')

        self.linear_speed   = float(_p('linear_speed'))
        self.kp             = float(_p('kp'))
        self.max_angular    = float(_p('max_angular'))
        self.search_linear  = float(_p('search_linear'))
        self.search_angular = float(_p('search_angular'))
        self.roi_y_start    = float(_p('roi_y_start'))
        self.min_area       = int(_p('min_area'))
        self.h_low   = int(_p('h_low'));  self.s_low  = int(_p('s_low'));  self.v_low  = int(_p('v_low'))
        self.h_high  = int(_p('h_high')); self.s_high = int(_p('s_high')); self.v_high = int(_p('v_high'))

        self.red_h1_low   = int(_p('red_h1_low'));   self.red_h1_high = int(_p('red_h1_high'))
        self.red_h2_low   = int(_p('red_h2_low'));   self.red_h2_high = int(_p('red_h2_high'))
        self.red_s_low    = int(_p('red_s_low'))
        self.red_v_low    = int(_p('red_v_low'))
        self.red_min_area = int(_p('red_min_area'))

        self.tof_delta      = float(_p('tof_delta_threshold'))
        self.tof_win_size   = int(_p('tof_baseline_window'))
        self.tof_confirm    = int(_p('tof_confirm_frames'))
        self.tof_alpha      = float(_p('tof_filter_alpha'))

        self.approach_speed      = float(_p('approach_speed'))
        self.approach_kp_ang     = float(_p('approach_kp_ang'))
        self.approach_max_ang    = float(_p('approach_max_ang'))
        self.approach_stop_area  = float(_p('approach_stop_area'))
        self.approach_stop_tof   = float(_p('approach_stop_front_tof'))
        self.approach_timeout    = float(_p('approach_timeout_sec'))
        self.approach_lost_limit = int(_p('approach_lost_frames'))
        self.front_hard_stop     = float(_p('front_hard_stop_dist'))

        self.scan_timeout       = float(_p('scan_timeout_sec'))
        self.scan_after_timeout = float(_p('scan_after_timeout_sec'))

        self.turn_angle     = float(_p('turn_angle_deg'))
        self.turn_tolerance = float(_p('turn_tolerance_deg'))
        self.turn_kp        = float(_p('turn_kp'))
        self.turn_min_speed = float(_p('turn_min_speed'))
        self.turn_max_speed = float(_p('turn_max_speed'))
        self.turn_timeout   = float(_p('turn_timeout_sec'))

        self.reverse_speed    = float(_p('reverse_speed'))
        self.reverse_distance = float(_p('reverse_distance'))
        self.reverse_time     = (self.reverse_distance / self.reverse_speed
                                 if self.reverse_speed > 1e-6 else 3.0)

        self.target_box_count   = int(_p('target_box_count'))
        self.pick_action_name   = str(_p('pick_action_name'))
        self.goto_line_speed    = float(_p('goto_line_speed'))
        self.goto_line_timeout  = float(_p('goto_line_timeout_sec'))
        self.resume_fwd_time    = float(_p('resume_forward_time'))
        self.debug              = bool(_p('debug'))

        # ── ROS interfaces ───────────────────────────────────────────────────
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_cb, qos_profile_sensor_data
        )
        self.left_sub = self.create_subscription(
            Range, self.left_range_topic, self._left_range_cb, qos_profile_sensor_data
        )
        self.right_sub = self.create_subscription(
            Range, self.right_range_topic, self._right_range_cb, qos_profile_sensor_data
        )
        self.front_sub = self.create_subscription(
            Range, self.front_range_topic, self._front_range_cb, qos_profile_sensor_data
        )
        self.gyro_sub = self.create_subscription(
            Float32, self.gyro_angle_topic, self._gyro_cb, qos_profile_sensor_data
        )

        self.cmd_pub    = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.pick_client = ActionClient(self, PickBox, self.pick_action_name)
        self._pick_server_ready = False
        self._server_check_timer = self.create_timer(1.0, self._check_pick_server)

        # ── FSM ──────────────────────────────────────────────────────────────
        self.state            = self.ST_SCAN
        self._state_t         = self.get_clock().now()

        # ── Sensing ──────────────────────────────────────────────────────────
        self.left_range  = math.inf
        self.right_range = math.inf
        self.front_range = math.inf

        # ── ToF change detection ─────────────────────────────────────────────
        self._left_win  = collections.deque(maxlen=self.tof_win_size)
        self._right_win = collections.deque(maxlen=self.tof_win_size)
        self._left_cnt  = 0   # consecutive trigger counter
        self._right_cnt = 0
        # Per-side suppression after a pick (avoid immediate re-trigger)
        self._suppress_side  = None
        self._suppress_until = None

        # ── Gyro turn ────────────────────────────────────────────────────────
        self.current_yaw_deg = None
        self.gyro_ready      = False
        self._turn_target    = None
        self._turn_next      = None
        self._turn_start_t   = None
        self._turn_dir       = 0.0   # +1 = CCW/left, -1 = CW/right

        # ── Box tracking ─────────────────────────────────────────────────────
        self.active_box_side   = None   # 'LEFT', 'RIGHT', or 'FRONT'
        self.box_count         = 0
        self._approach_start_t = None
        self._approach_lost_n  = 0

        # ── Pick action ──────────────────────────────────────────────────────
        self._pick_result_ready   = False
        self._pick_result_success = False
        self._pick_in_progress    = False

        # ── Debug ─────────────────────────────────────────────────────────────
        if self.debug:
            cv2.namedWindow('task2', cv2.WINDOW_NORMAL)
            cv2.namedWindow('red_mask', cv2.WINDOW_NORMAL)

        self.get_logger().info(
            f'Task2Collector started | initial state: {self.state}\n'
            f'  tof_delta={self.tof_delta:.2f}m  confirm={self.tof_confirm}fr  '
            f'window={self.tof_win_size}fr\n'
            f'  approach_stop_area={self.approach_stop_area:.0f}px²  '
            f'approach_stop_tof={self.approach_stop_tof:.2f}m\n'
            f'  target_boxes={self.target_box_count}  debug={self.debug}'
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Utilities
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _clamp(x, lo, hi):
        return max(lo, min(hi, x))

    @staticmethod
    def _norm_deg(a):
        while a > 180.0:
            a -= 360.0
        while a < -180.0:
            a += 360.0
        return a

    def _valid(self, x):
        return not (math.isinf(x) or math.isnan(x)) and x > 0.0

    def _elapsed(self, t):
        return (self.get_clock().now() - t).nanoseconds * 1e-9

    def _set_state(self, new_state):
        self.get_logger().info(f'FSM  {self.state} → {new_state}')
        self.state    = new_state
        self._state_t = self.get_clock().now()

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _drive(self, vx: float, wz: float):
        t = Twist()
        t.linear.x  = float(vx)
        t.angular.z = float(wz)
        self.cmd_pub.publish(t)

    # ═══════════════════════════════════════════════════════════════════════════
    # Sensor callbacks
    # ═══════════════════════════════════════════════════════════════════════════

    def _lp(self, prev, cur):
        """Exponential moving average."""
        if not self._valid(prev):
            return cur
        return self.tof_alpha * cur + (1.0 - self.tof_alpha) * prev

    def _left_range_cb(self, msg: Range):
        raw = float(msg.range)
        if self._valid(raw):
            self.left_range = self._lp(self.left_range, raw)

    def _right_range_cb(self, msg: Range):
        raw = float(msg.range)
        if self._valid(raw):
            self.right_range = self._lp(self.right_range, raw)

    def _front_range_cb(self, msg: Range):
        raw = float(msg.range)
        if self._valid(raw):
            self.front_range = self._lp(self.front_range, raw)

    def _gyro_cb(self, msg: Float32):
        self.current_yaw_deg = self._norm_deg(float(msg.data))
        self.gyro_ready = True

    def _check_pick_server(self):
        if not self._pick_server_ready and self.pick_client.server_is_ready():
            self._pick_server_ready = True
            self.get_logger().info('/pick_box action server is ready.')

    # ═══════════════════════════════════════════════════════════════════════════
    # Vision helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _red_mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lo1 = np.array([self.red_h1_low, self.red_s_low, self.red_v_low], np.uint8)
        hi1 = np.array([self.red_h1_high, 255, 255], np.uint8)
        lo2 = np.array([self.red_h2_low, self.red_s_low, self.red_v_low], np.uint8)
        hi2 = np.array([self.red_h2_high, 255, 255], np.uint8)
        m = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1), cv2.inRange(hsv, lo2, hi2))
        k = np.ones((5, 5), np.uint8)
        m = cv2.GaussianBlur(m, (5, 5), 0)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        return m

    def _detect_red_box(self, frame):
        """Returns (found, cx, cy, area, bbox_xywh, mask)."""
        mask = self._red_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_c, best_a = None, 0.0
        for c in contours:
            a = cv2.contourArea(c)
            if a >= self.red_min_area and a > best_a:
                best_a, best_c = a, c
        if best_c is None:
            return False, 0, 0, 0.0, None, mask
        x, y, w, h = cv2.boundingRect(best_c)
        M = cv2.moments(best_c)
        if M['m00'] <= 0:
            return False, 0, 0, 0.0, None, mask
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return True, cx, cy, float(best_a), (x, y, w, h), mask

    # ═══════════════════════════════════════════════════════════════════════════
    # ToF change detection
    # ═══════════════════════════════════════════════════════════════════════════

    def _tof_baseline(self, side: str) -> float:
        win = self._left_win if side == 'LEFT' else self._right_win
        return sum(win) / len(win) if win else math.inf

    def _suppress_active(self, side: str) -> bool:
        if self._suppress_side != side or self._suppress_until is None:
            return False
        if self.get_clock().now() < self._suppress_until:
            return True
        self._suppress_side  = None
        self._suppress_until = None
        return False

    def _start_suppress(self, side: str, secs: float = 4.0):
        self._suppress_side  = side
        self._suppress_until = self.get_clock().now() + Duration(seconds=secs)
        self.get_logger().info(f'Suppressing {side} ToF trigger for {secs:.1f}s.')

    def _update_tof_baselines(self):
        """Feed filtered readings into rolling-baseline windows."""
        if self._valid(self.left_range):
            self._left_win.append(self.left_range)
        if self._valid(self.right_range):
            self._right_win.append(self.right_range)

    def _check_tof_change(self):
        """
        Returns 'LEFT', 'RIGHT', or None.

        Triggered when the current filtered ToF reading on either side drops
        more than tof_delta below the rolling-window baseline for
        tof_confirm_frames consecutive frames.
        """
        for side in ('LEFT', 'RIGHT'):
            if self._suppress_active(side):
                setattr(self, f'_{"left" if side=="LEFT" else "right"}_cnt', 0)
                continue

            win  = self._left_win  if side == 'LEFT' else self._right_win
            cur  = self.left_range  if side == 'LEFT' else self.right_range
            attr = '_left_cnt'     if side == 'LEFT' else '_right_cnt'

            if len(win) < self.tof_win_size or not self._valid(cur):
                continue

            baseline = sum(win) / len(win)
            if baseline - cur > self.tof_delta:
                setattr(self, attr, getattr(self, attr) + 1)
            else:
                setattr(self, attr, 0)

            if getattr(self, attr) >= self.tof_confirm:
                self.get_logger().info(
                    f'ToF change on {side}: baseline={baseline:.3f}m '
                    f'current={cur:.3f}m drop={baseline-cur:.3f}m'
                )
                return side
        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # Gyro turn helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _start_turn(self, dir_sign: float, angle_deg: float, next_state: str):
        """
        dir_sign: +1 = CCW (left), -1 = CW (right).
        angle_deg: magnitude of desired rotation.
        """
        if not self.gyro_ready:
            self.get_logger().warn('Gyro not ready — skipping turn, going straight to next state.')
            self._set_state(next_state)
            return
        self._turn_dir    = dir_sign
        self._turn_target = self._norm_deg(self.current_yaw_deg + dir_sign * angle_deg)
        self._turn_next   = next_state
        self._turn_start_t = self.get_clock().now()
        self.get_logger().info(
            f'Turn {dir_sign:+.0f}×{angle_deg:.0f}° → target={self._turn_target:.1f}°'
            f' (current={self.current_yaw_deg:.1f}°)'
        )

    def _run_turn(self) -> bool:
        """Run one tick of the gyro-turn controller. Returns True when done."""
        err = self._norm_deg(self._turn_target - self.current_yaw_deg)

        if abs(err) <= self.turn_tolerance:
            self._stop()
            return True

        if self._elapsed(self._turn_start_t) > self.turn_timeout:
            self.get_logger().warn('Turn timeout — forcing next state.')
            self._stop()
            return True

        spd = self._clamp(self.turn_kp * abs(err), self.turn_min_speed, self.turn_max_speed)
        self._drive(0.0, math.copysign(spd, err))
        return False

    # ═══════════════════════════════════════════════════════════════════════════
    # Pick action client
    # ═══════════════════════════════════════════════════════════════════════════

    def _send_pick(self, side: str):
        """Send a /pick_box goal asynchronously."""
        if not self._pick_server_ready:
            self.get_logger().warn('Pick server not ready yet — trying anyway.')
        goal = PickBox.Goal()
        goal.side = side if side in ('LEFT', 'RIGHT') else 'LEFT'
        fut = self.pick_client.send_goal_async(goal)
        fut.add_done_callback(self._goal_response_cb)
        self._pick_in_progress = True

    def _goal_response_cb(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('Pick goal rejected.')
            self._pick_result_ready   = True
            self._pick_result_success = False
            self._pick_in_progress    = False
            return
        gh.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        result = future.result().result
        self._pick_result_success = result.success
        self._pick_result_ready   = True
        self._pick_in_progress    = False
        self.get_logger().info(f'Pick result: success={result.success} | {result.message}')

    # ═══════════════════════════════════════════════════════════════════════════
    # Main image callback (drives FSM)
    # ═══════════════════════════════════════════════════════════════════════════

    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        H, W  = frame.shape[:2]
        vis   = frame.copy() if self.debug else None

        # ── dispatch ─────────────────────────────────────────────────────────
        if self.state == self.ST_SCAN:
            self._do_scan(frame, W, H, vis)

        elif self.state == self.ST_GOTO_LINE:
            self._do_goto_line(frame, W, H, vis)

        elif self.state == self.ST_FOLLOW:
            self._do_follow(frame, W, H, vis)

        elif self.state == self.ST_TURN_TO_BOX:
            if not self.gyro_ready:
                self._stop()
            elif self._run_turn():
                self._approach_start_t = self.get_clock().now()
                self._approach_lost_n  = 0
                self._set_state(self._turn_next)

        elif self.state == self.ST_APPROACH:
            self._do_approach(frame, W, H, vis)

        elif self.state == self.ST_REQ_PICK:
            self._stop()
            self._pick_result_ready   = False
            self._pick_result_success = False
            self._send_pick(self.active_box_side)
            self._set_state(self.ST_WAIT_PICK)

        elif self.state == self.ST_WAIT_PICK:
            self._stop()
            if self._pick_result_ready:
                if self._pick_result_success:
                    self.box_count += 1
                    self.get_logger().info(
                        f'Box collected: {self.box_count}/{self.target_box_count}'
                    )
                else:
                    self.get_logger().warn('Pick reported failure — counting anyway.')
                    self.box_count += 1

                if self.box_count >= self.target_box_count:
                    self._set_state(self.ST_DONE)
                else:
                    # Suppress re-detection of the same side for a while
                    if self.active_box_side in ('LEFT', 'RIGHT'):
                        self._start_suppress(self.active_box_side, secs=5.0)
                    self._set_state(self.ST_REVERSE)

        elif self.state == self.ST_REVERSE:
            self._do_reverse()

        elif self.state == self.ST_TURN_BACK:
            if not self.gyro_ready:
                self._stop()
            elif self._run_turn():
                self._set_state(self._turn_next)

        elif self.state == self.ST_SCAN_AFTER:
            self._do_scan_after(frame, W, H, vis)

        elif self.state == self.ST_RESUME:
            self._do_resume(frame, W, H, vis)

        elif self.state == self.ST_DONE:
            self._stop()
            out = String()
            out.data = 'DONE'
            self.status_pub.publish(out)

        # ── debug overlay ─────────────────────────────────────────────────────
        if self.debug and vis is not None:
            lbls = [
                f'State: {self.state}',
                f'Boxes: {self.box_count}/{self.target_box_count}',
                f'L={self.left_range:.2f}m  F={self.front_range:.2f}m  R={self.right_range:.2f}m',
                (f'L_base={self._tof_baseline("LEFT"):.2f}m  '
                 f'R_base={self._tof_baseline("RIGHT"):.2f}m'),
            ]
            for i, lbl in enumerate(lbls):
                cv2.putText(vis, lbl, (10, 25 + 22 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)
            cv2.imshow('task2', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._stop()
                rclpy.shutdown()

    # ═══════════════════════════════════════════════════════════════════════════
    # State handlers
    # ═══════════════════════════════════════════════════════════════════════════

    # ── 1. SCAN_FOR_BOXES ────────────────────────────────────────────────────

    def _do_scan(self, frame, W, H, vis):
        """
        Stand still, scan camera for a red box straight ahead.
        If found → approach directly (no turn).
        If timeout → go look for white line.
        """
        found, cx, cy, area, bbox, mask = self._detect_red_box(frame)
        elapsed = self._elapsed(self._state_t)

        if self.debug and mask is not None:
            cv2.imshow('red_mask', mask)

        if found:
            self.get_logger().info(
                f'SCAN: Box straight ahead — area={area:.0f}px² cx={cx}. Approaching.'
            )
            self.active_box_side   = 'FRONT'
            self._approach_start_t = self.get_clock().now()
            self._approach_lost_n  = 0
            self._set_state(self.ST_APPROACH)
        elif elapsed >= self.scan_timeout:
            self.get_logger().info(
                f'SCAN: No box found after {elapsed:.1f}s. Going to find white line.'
            )
            self._set_state(self.ST_GOTO_LINE)
        else:
            self._stop()
            if self.debug and vis is not None:
                cv2.putText(vis,
                            f'Scanning... {elapsed:.1f}/{self.scan_timeout:.1f}s',
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # ── 2. GOTO_LINE ─────────────────────────────────────────────────────────

    def _do_goto_line(self, frame, W, H, vis):
        """
        Drive forward slowly until the white line appears in the camera ROI.
        """
        elapsed = self._elapsed(self._state_t)
        if elapsed >= self.goto_line_timeout:
            self.get_logger().warn(
                f'GOTO_LINE: {elapsed:.1f}s timeout — starting FOLLOW_LINE anyway.'
            )
            self._init_follow()
            self._set_state(self.ST_FOLLOW)
            return

        y0   = int(H * self.roi_y_start)
        roi  = frame[y0:H, :]
        mask = build_white_mask(roi,
                                self.h_low, self.s_low, self.v_low,
                                self.h_high, self.s_high, self.v_high)
        area = cv2.moments(mask)['m00']

        if area >= self.min_area:
            self.get_logger().info('GOTO_LINE: White line detected — starting FOLLOW_LINE.')
            self._init_follow()
            self._set_state(self.ST_FOLLOW)
        else:
            self._drive(self.goto_line_speed, 0.0)

    def _init_follow(self):
        """Reset baselines and counters before entering FOLLOW_LINE."""
        self._left_win.clear()
        self._right_win.clear()
        self._left_cnt  = 0
        self._right_cnt = 0

    # ── 3. FOLLOW_LINE ───────────────────────────────────────────────────────

    def _do_follow(self, frame, W, H, vis):
        """
        Follow white line with P-control.
        Simultaneously feed side-ToF readings into baseline windows.
        Trigger box-approach when a significant distance drop is detected.
        """
        # Feed baselines (only meaningful while straight-line following)
        self._update_tof_baselines()

        # Check for significant ToF change on either side
        detected = self._check_tof_change()
        if detected:
            self._stop()
            self.active_box_side = detected
            dir_sign = 1.0 if detected == 'LEFT' else -1.0
            self._start_turn(dir_sign, self.turn_angle, self.ST_APPROACH)
            self._set_state(self.ST_TURN_TO_BOX)
            return

        # White-line P-control
        y0   = int(H * self.roi_y_start)
        roi  = frame[y0:H, :]
        mask = build_white_mask(roi,
                                self.h_low, self.s_low, self.v_low,
                                self.h_high, self.s_high, self.v_high)
        M    = cv2.moments(mask)
        area = M['m00']

        if area >= self.min_area:
            cx  = int(M['m10'] / area)
            err = float(cx - W // 2)
            ang = self._clamp(-self.kp * err, -self.max_angular, self.max_angular)
            self._drive(self.linear_speed, ang)
            if self.debug and vis is not None:
                cy_roi = int(H * (1.0 - self.roi_y_start) / 2)
                cv2.circle(vis, (cx, y0 + cy_roi), 8, (0, 255, 0), -1)
                cv2.putText(vis, f'line_err={err:.0f}', (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            # Line lost: gentle forward + angular search
            self._drive(self.search_linear, self.search_angular)

    # ── 5. CENTER_APPROACH ───────────────────────────────────────────────────

    def _do_approach(self, frame, W, H, vis):
        """
        Camera-centre the red box and creep forward until at pick distance.

        Straight-approach guarantee:
            Forward speed is multiplied by (1 - angular_correction_fraction).
            When angular error is large the robot corrects first and advances
            slowly — ensuring it arrives straight-on.
        """
        if self._approach_start_t is None:
            self._approach_start_t = self.get_clock().now()

        found, cx, cy, area, bbox, mask = self._detect_red_box(frame)

        if self.debug and mask is not None:
            cv2.imshow('red_mask', mask)

        # ── Safety hard stop ─────────────────────────────────────────────────
        if self._valid(self.front_range) and self.front_range < self.front_hard_stop:
            self.get_logger().warn(
                f'APPROACH: Hard stop at {self.front_range:.3f}m — requesting pick.'
            )
            self._stop()
            self._set_state(self.ST_REQ_PICK)
            return

        # ── Approach timeout ─────────────────────────────────────────────────
        if self._elapsed(self._approach_start_t) > self.approach_timeout:
            self.get_logger().warn('APPROACH: Timeout — requesting pick at current position.')
            self._stop()
            self._set_state(self.ST_REQ_PICK)
            return

        if found:
            self._approach_lost_n = 0

            # ── Stop conditions ───────────────────────────────────────────────
            area_ok = area >= self.approach_stop_area
            tof_ok  = self._valid(self.front_range) and self.front_range <= self.approach_stop_tof

            if area_ok or tof_ok:
                reason = f'area={area:.0f}px²' if area_ok else f'tof={self.front_range:.3f}m'
                self.get_logger().info(f'APPROACH: At pick distance ({reason}) — stopping.')
                self._stop()
                self._set_state(self.ST_REQ_PICK)
                return

            # ── P-control centering + speed scaling for straight arrival ─────
            err_x = float(cx - W // 2)
            ang   = self._clamp(-self.approach_kp_ang * err_x,
                                -self.approach_max_ang, self.approach_max_ang)

            # Reduce forward speed proportional to angular correction magnitude.
            # At max angular correction: fwd = approach_speed * 0.2 (creep only).
            # At zero angular error:    fwd = approach_speed       (full creep).
            ang_frac = abs(ang) / self.approach_max_ang          # 0..1
            fwd = self.approach_speed * (1.0 - self._clamp(ang_frac * 0.8, 0.0, 0.8))
            self._drive(fwd, ang)

            if self.debug and vis is not None:
                cv2.circle(vis, (cx, cy), 12, (0, 0, 255), -1)
                cv2.line(vis, (W // 2, 0), (W // 2, H), (180, 180, 180), 1)
                cv2.putText(vis,
                            f'err={err_x:.0f}px  area={area:.0f}  fwd={fwd:.3f}  tof={self.front_range:.3f}',
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 100, 255), 2)
        else:
            # Box not visible this frame
            self._approach_lost_n += 1

            if self._approach_lost_n >= self.approach_lost_limit:
                self.get_logger().warn(
                    f'APPROACH: Box lost for {self._approach_lost_n} frames — aborting.'
                )
                self._stop()
                # Turn back and scan
                self._turn_toward_line_then_scan()
            else:
                # Hold position while waiting for box to reappear
                self._stop()

    def _turn_toward_line_then_scan(self):
        """After failed approach: turn back toward line heading, then scan."""
        if self.active_box_side == 'FRONT':
            # No turn needed
            self._set_state(self.ST_SCAN_AFTER)
        else:
            back_dir = -1.0 if self.active_box_side == 'LEFT' else 1.0
            self._start_turn(back_dir, self.turn_angle, self.ST_SCAN_AFTER)
            self._set_state(self.ST_TURN_BACK)

    # ── 8. REVERSE ───────────────────────────────────────────────────────────

    def _do_reverse(self):
        """
        Drive backward to return to the corridor / white-line position.
        After reverse_time seconds, initiate turn-back.
        """
        elapsed = self._elapsed(self._state_t)
        if elapsed >= self.reverse_time:
            self._stop()
            # Turn back toward original heading
            if self.active_box_side == 'FRONT':
                self._set_state(self.ST_SCAN_AFTER)
            else:
                back_dir = -1.0 if self.active_box_side == 'LEFT' else 1.0
                self._start_turn(back_dir, self.turn_angle, self.ST_SCAN_AFTER)
                self._set_state(self.ST_TURN_BACK)
        else:
            self._drive(-self.reverse_speed, 0.0)

    # ── 10. SCAN_AFTER_PICK ──────────────────────────────────────────────────

    def _do_scan_after(self, frame, W, H, vis):
        """
        Quick scan for another box after returning to line heading.
        Require a larger blob (more confident) before approaching.
        """
        found, cx, cy, area, bbox, mask = self._detect_red_box(frame)
        elapsed = self._elapsed(self._state_t)

        if self.debug and mask is not None:
            cv2.imshow('red_mask', mask)

        if found and area >= self.red_min_area * 3:
            self.get_logger().info(
                f'SCAN_AFTER: Another box found — area={area:.0f}px² cx={cx}. Approaching.'
            )
            self.active_box_side   = 'FRONT'
            self._approach_start_t = self.get_clock().now()
            self._approach_lost_n  = 0
            self._set_state(self.ST_APPROACH)
        elif elapsed >= self.scan_after_timeout:
            self.get_logger().info('SCAN_AFTER: No box. Resuming line following.')
            self._set_state(self.ST_RESUME)
        else:
            self._stop()

    # ── 11. RESUME_LINE ──────────────────────────────────────────────────────

    def _do_resume(self, frame, W, H, vis):
        """
        Drive forward until white line re-appears in camera ROI.
        Also resets ToF baselines so the next ToF change detection starts fresh.
        """
        elapsed = self._elapsed(self._state_t)

        y0   = int(H * self.roi_y_start)
        roi  = frame[y0:H, :]
        mask = build_white_mask(roi,
                                self.h_low, self.s_low, self.v_low,
                                self.h_high, self.s_high, self.v_high)
        area = cv2.moments(mask)['m00']

        if area >= self.min_area:
            self.get_logger().info('RESUME: White line found — returning to FOLLOW_LINE.')
            self._init_follow()
            self._set_state(self.ST_FOLLOW)
        elif elapsed >= self.resume_fwd_time:
            self.get_logger().warn(
                f'RESUME: {elapsed:.1f}s without finding line — starting FOLLOW_LINE anyway.'
            )
            self._init_follow()
            self._set_state(self.ST_FOLLOW)
        else:
            # Drive forward along line direction (line should be ahead)
            self._drive(self.linear_speed, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = Task2Collector()
    try:
        rclpy.spin(node)
    finally:
        node._stop()
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
