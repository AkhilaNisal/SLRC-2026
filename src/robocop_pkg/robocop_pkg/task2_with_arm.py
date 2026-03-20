#!/usr/bin/env python3

# starting pos in arena :translation 0.75 -1.05 0

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.action import ActionClient

from sensor_msgs.msg import Image, Range
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
import numpy as np

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

        # =========================
        # Motion control
        # =========================
        self.declare_parameter('forward_speed', 0.12)
        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('kp', 0.004)
        self.declare_parameter('max_angular', 1.2)

        self.declare_parameter('extra_forward_distance', 0.18)

        self.declare_parameter('turn_left_angular_speed', 0.8)
        self.declare_parameter('turn_left_90_time', 2.25)
        self.declare_parameter('post_turn_wait_time', 1.0)

        self.declare_parameter('search_linear', 0.04)
        self.declare_parameter('search_angular', 0.35)

        # =========================
        # White detection
        # =========================
        self.declare_parameter('roi_y_start', 0.60)
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
        self.declare_parameter('red_min_area', 600)
        self.declare_parameter('red_kp', 0.0045)
        self.declare_parameter('red_max_angular', 1.0)
        self.declare_parameter('red_forward_speed', 0.08)
        self.declare_parameter('red_search_angular', 0.35)
        self.declare_parameter('red_stop_area', 18000)
        self.declare_parameter('red_lost_frames_limit', 12)

        # =========================
        # Distance sensing / filter
        # =========================
        self.declare_parameter('range_filter_alpha', 0.2)
        self.declare_parameter('print_distances_every_frame', False)

        # =========================
        # Box detection while following line
        # =========================
        self.declare_parameter('box_detect_distance', 0.50)
        self.declare_parameter('box_detect_frames', 8)
 
        # =========================
        # Box visit maneuver
        # =========================
        self.declare_parameter('box_turn_angular_speed', 0.8)
        self.declare_parameter('box_turn_90_time', 2.25)
        self.declare_parameter('box_turn_180_time', 4.75)
        self.declare_parameter('box_stop_distance', 0.18)
        self.declare_parameter('box_stop_frames', 5)
        self.declare_parameter('box_return_speed', 0.12)

        # =========================
        # Action client / pickup behavior
        # =========================
        self.declare_parameter('pick_action_name', '/pick_box')
        self.declare_parameter('pick_retry_limit', 2)
        self.declare_parameter('pick_goal_send_once', True)

        # =========================
        # Task 2 finish behavior
        # =========================
        self.declare_parameter('target_box_count', 6)
        self.declare_parameter('task2_finish_wall_distance', 0.15)
        self.declare_parameter('task2_finish_wall_frames', 3)
        self.declare_parameter('task2_finish_forward_speed', 0.08)

        # =========================
        # Read params
        # =========================
        self.image_topic = self.get_parameter('image_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.left_range_topic = self.get_parameter('left_range_topic').value
        self.right_range_topic = self.get_parameter('right_range_topic').value
        self.front_range_topic = self.get_parameter('front_range_topic').value
        self.task2_status_topic = self.get_parameter('task2_status_topic').value

        self.forward_speed = float(self.get_parameter('forward_speed').value)
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.kp = float(self.get_parameter('kp').value)
        self.max_angular = float(self.get_parameter('max_angular').value)

        self.extra_forward_distance = float(self.get_parameter('extra_forward_distance').value)
        self.extra_forward_time = (
            self.extra_forward_distance / self.forward_speed
            if self.forward_speed > 1e-6 else 0.0
        )

        self.turn_left_angular_speed = float(self.get_parameter('turn_left_angular_speed').value)
        self.turn_left_90_time = float(self.get_parameter('turn_left_90_time').value)
        self.post_turn_wait_time = float(self.get_parameter('post_turn_wait_time').value)

        self.search_linear = float(self.get_parameter('search_linear').value)
        self.search_angular = float(self.get_parameter('search_angular').value)

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
        self.red_forward_speed = float(self.get_parameter('red_forward_speed').value)
        self.red_search_angular = float(self.get_parameter('red_search_angular').value)
        self.red_stop_area = int(self.get_parameter('red_stop_area').value)
        self.red_lost_frames_limit = int(self.get_parameter('red_lost_frames_limit').value)

        self.range_filter_alpha = float(self.get_parameter('range_filter_alpha').value)
        self.print_distances_every_frame = bool(
            self.get_parameter('print_distances_every_frame').value
        )

        self.box_detect_distance = float(self.get_parameter('box_detect_distance').value)
        self.box_detect_frames = int(self.get_parameter('box_detect_frames').value)

        self.box_turn_angular_speed = float(self.get_parameter('box_turn_angular_speed').value)
        self.box_turn_90_time = float(self.get_parameter('box_turn_90_time').value)
        self.box_turn_180_time = float(self.get_parameter('box_turn_180_time').value)
        self.box_stop_distance = float(self.get_parameter('box_stop_distance').value)
        self.box_stop_frames = int(self.get_parameter('box_stop_frames').value)
        self.box_return_speed = float(self.get_parameter('box_return_speed').value)

        self.pick_action_name = str(self.get_parameter('pick_action_name').value)
        self.pick_retry_limit = int(self.get_parameter('pick_retry_limit').value)
        self.pick_goal_send_once = bool(self.get_parameter('pick_goal_send_once').value)

        self.target_box_count = int(self.get_parameter('target_box_count').value)
        self.task2_finish_wall_distance = float(self.get_parameter('task2_finish_wall_distance').value)
        self.task2_finish_wall_frames = int(self.get_parameter('task2_finish_wall_frames').value)
        self.task2_finish_forward_speed = float(self.get_parameter('task2_finish_forward_speed').value)

        # =========================
        # ROS interfaces
        # =========================
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

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.task_status_pub = self.create_publisher(String, self.task2_status_topic, 10)

        self.pick_client = ActionClient(self, PickBox, self.pick_action_name)

        self.pick_server_ready_logged = False
        self.server_check_timer = self.create_timer(0.5, self.check_pick_server)

        # =========================
        # States
        # =========================
        self.STATE_LINE_CROSS_APPROACH = 'LINE_CROSS_APPROACH'
        self.STATE_LINE_CROSS_WAIT_DISAPPEAR = 'LINE_CROSS_WAIT_DISAPPEAR'
        self.STATE_LINE_CROSS_EXTRA_FORWARD = 'LINE_CROSS_EXTRA_FORWARD'
        self.STATE_LINE_CROSS_TURN = 'LINE_CROSS_TURN'
        self.STATE_LINE_CROSS_POST_WAIT = 'LINE_CROSS_POST_WAIT'

        self.STATE_FOLLOW_LINE = 'FOLLOW_LINE'

        self.STATE_BOX_TURN_TO_BOX = 'BOX_TURN_TO_BOX'
        self.STATE_BOX_DRIVE_TO_BOX = 'BOX_DRIVE_TO_BOX'
        self.STATE_BOX_REQUEST_PICK = 'BOX_REQUEST_PICK'
        self.STATE_BOX_WAIT_PICK_RESULT = 'BOX_WAIT_PICK_RESULT'
        self.STATE_BOX_PICK_FAILED = 'BOX_PICK_FAILED'
        self.STATE_BOX_TURN_BACK_180 = 'BOX_TURN_BACK_180'
        self.STATE_BOX_RETURN_TO_LINE = 'BOX_RETURN_TO_LINE'

        self.STATE_TASK2_DONE = 'TASK2_DONE'

        self.state = self.STATE_LINE_CROSS_APPROACH

        # line-cross reusable context
        self.line_cross_turn_direction = +1.0
        self.line_cross_next_state = self.STATE_FOLLOW_LINE
        self.line_seen = False
        self.line_gone_counter = 0
        self.extra_forward_start_time = None
        self.turn_start_time = None
        self.post_turn_wait_start_time = None
        self.line_cross_speed = self.forward_speed

        # box visit state
        self.active_box_side = None
        self.box_stop_counter = 0
        self.left_box_count = 0
        self.right_box_count = 0
        self.red_lost_counter = 0
        self.boxes_completed = 0

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
        self.left_detect_counter = 0
        self.right_detect_counter = 0
        self.finish_wall_counter = 0
        self.task2_done = False
        self.task2_done_published = False

        # debug
        self.frame_count = 0
        self.last_log_time = self.get_clock().now()

        cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("bottom_mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("red_mask", cv2.WINDOW_NORMAL)

        self.configure_line_cross_sequence(
            speed=self.forward_speed,
            turn_direction=+1.0,
            next_state=self.STATE_FOLLOW_LINE
        )

        self.get_logger().info("Started task2_with_arm with front-wall finish logic.")

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

    def valid_range(self, x: float) -> bool:
        return not math.isinf(x) and not math.isnan(x) and x > 0.0

    def display_range(self, x: float) -> str:
        if not self.measurement_started:
            return "OFF"
        return self.fmt_range(x)

    def side_sign(self, side: str) -> float:
        return 1.0 if side == 'LEFT' else -1.0

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
        self.left_range = self.low_pass_filter(self.left_range, raw, self.range_filter_alpha)

    def right_range_cb(self, msg: Range):
        raw = float(msg.range)
        self.right_range_raw = raw
        self.right_range = self.low_pass_filter(self.right_range, raw, self.range_filter_alpha)

    def front_range_cb(self, msg: Range):
        raw = float(msg.range)
        self.front_range_raw = raw
        self.front_range = self.low_pass_filter(self.front_range, raw, self.range_filter_alpha)

    def build_white_mask(self, bgr_img):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_low, self.s_low, self.v_low], dtype=np.uint8)
        upper = np.array([self.h_high, self.s_high, self.v_high], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return mask

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
        self.get_logger().info("Started box detection during white-line following.")

    def reset_box_detection_counters(self):
        self.left_detect_counter = 0
        self.right_detect_counter = 0

    def choose_box_side(self):
        if not self.measurement_started or self.state != self.STATE_FOLLOW_LINE:
            self.reset_box_detection_counters()
            return None

        left_hit = self.valid_range(self.left_range) and self.left_range < self.box_detect_distance
        right_hit = self.valid_range(self.right_range) and self.right_range < self.box_detect_distance

        self.left_detect_counter = self.left_detect_counter + 1 if left_hit else 0
        self.right_detect_counter = self.right_detect_counter + 1 if right_hit else 0

        left_ready = self.left_detect_counter >= self.box_detect_frames
        right_ready = self.right_detect_counter >= self.box_detect_frames

        if not left_ready and not right_ready:
            return None

        if left_ready and right_ready:
            side = 'LEFT' if self.left_range <= self.right_range else 'RIGHT'
        elif left_ready:
            side = 'LEFT'
        else:
            side = 'RIGHT'

        self.reset_box_detection_counters()
        return side

    def configure_line_cross_sequence(self, speed: float, turn_direction: float, next_state: str):
        self.line_cross_speed = speed
        self.line_cross_turn_direction = turn_direction
        self.line_cross_next_state = next_state
        self.line_seen = False
        self.line_gone_counter = 0
        self.extra_forward_start_time = None
        self.turn_start_time = None
        self.post_turn_wait_start_time = None
        self.state = self.STATE_LINE_CROSS_APPROACH

    def start_box_detour(self, side: str):
        self.active_box_side = side
        self.box_stop_counter = 0
        self.red_lost_counter = 0
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

        self.state = self.STATE_BOX_TURN_TO_BOX
        self.turn_start_time = self.get_clock().now()
        self.get_logger().info(f"{side} box detected. count={count}. Starting detour toward box.")

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
                self.state = self.STATE_LINE_CROSS_TURN
                self.turn_start_time = self.get_clock().now()
                self.get_logger().info("Extra forward done. Starting turn.")

        elif self.state == self.STATE_LINE_CROSS_TURN:
            twist.linear.x = 0.0
            twist.angular.z = self.line_cross_turn_direction * self.turn_left_angular_speed

            elapsed = (self.get_clock().now() - self.turn_start_time).nanoseconds / 1e9
            if elapsed >= self.turn_left_90_time:
                self.state = self.STATE_LINE_CROSS_POST_WAIT
                self.post_turn_wait_start_time = self.get_clock().now()
                self.get_logger().info(
                    f"Turn complete. Waiting {self.post_turn_wait_time:.2f}s."
                )

        elif self.state == self.STATE_LINE_CROSS_POST_WAIT:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

            elapsed = (self.get_clock().now() - self.post_turn_wait_start_time).nanoseconds / 1e9
            if elapsed >= self.post_turn_wait_time:
                self.state = self.line_cross_next_state
                if self.state == self.STATE_FOLLOW_LINE:
                    self.start_measurement()
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

        if self.state in {
            self.STATE_LINE_CROSS_APPROACH,
            self.STATE_LINE_CROSS_WAIT_DISAPPEAR,
            self.STATE_LINE_CROSS_EXTRA_FORWARD,
            self.STATE_LINE_CROSS_TURN,
            self.STATE_LINE_CROSS_POST_WAIT,
        }:
            self.run_line_cross_sequence(bottom_area, twist)

        elif self.state == self.STATE_FOLLOW_LINE:
            # After all boxes are complete, prioritize final wall stopping.
            if self.boxes_completed >= self.target_box_count:
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
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.start_box_detour(side)

        elif self.state == self.STATE_BOX_TURN_TO_BOX:
            twist.linear.x = 0.0
            twist.angular.z = self.side_sign(self.active_box_side) * self.box_turn_angular_speed

            elapsed = (self.get_clock().now() - self.turn_start_time).nanoseconds / 1e9
            if elapsed >= self.box_turn_90_time:
                self.state = self.STATE_BOX_DRIVE_TO_BOX
                self.box_stop_counter = 0
                self.red_lost_counter = 0
                self.get_logger().info(f"Turned toward {self.active_box_side} box. Tracking red box.")

        elif self.state == self.STATE_BOX_DRIVE_TO_BOX:
            if red_found:
                self.red_lost_counter = 0

                error = float(red_cx - (w // 2))
                ang = -self.red_kp * error
                ang = self.clamp(ang, -self.red_max_angular, self.red_max_angular)

                twist.linear.x = self.red_forward_speed
                twist.angular.z = ang

                side_dist = self.current_side_range()
                near_by_range = self.valid_range(side_dist) and side_dist <= self.box_stop_distance
                near_by_area = red_area >= self.red_stop_area

                if near_by_range or near_by_area:
                    self.box_stop_counter += 1
                else:
                    self.box_stop_counter = 0

                if self.box_stop_counter >= self.box_stop_frames:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.state = self.STATE_BOX_REQUEST_PICK
                    self.get_logger().info(
                        f"Reached {self.active_box_side} red box. "
                        f"range={self.fmt_range(side_dist)} area={int(red_area)}. "
                        f"Stopping and requesting pick."
                    )
            else:
                self.red_lost_counter += 1
                twist.linear.x = 0.0
                twist.angular.z = self.side_sign(self.active_box_side) * self.red_search_angular

                if self.red_lost_counter > self.red_lost_frames_limit:
                    self.get_logger().info("Red box lost. Rotating slowly to reacquire target.")

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
                    self.state = self.STATE_BOX_TURN_BACK_180
                    self.turn_start_time = self.get_clock().now()
                    self.get_logger().info(
                        f"Pick success. boxes_completed={self.boxes_completed}. Turning back 180 degrees."
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
                self.state = self.STATE_BOX_REQUEST_PICK
                self.get_logger().warn(
                    f"Retrying pick action: attempt {self.pick_retry_count}/{self.pick_retry_limit}"
                )
            else:
                if not self.pick_failed_latched:
                    self.get_logger().warn(
                        "Pick failed and retry limit reached. Staying stopped at box."
                    )
                    self.pick_failed_latched = True
                twist.linear.x = 0.0
                twist.angular.z = 0.0

        elif self.state == self.STATE_BOX_TURN_BACK_180:
            twist.linear.x = 0.0
            twist.angular.z = self.side_sign(self.active_box_side) * self.box_turn_angular_speed

            elapsed = (self.get_clock().now() - self.turn_start_time).nanoseconds / 1e9
            if elapsed >= self.box_turn_180_time:
                self.state = self.STATE_BOX_RETURN_TO_LINE
                self.get_logger().info("Turned back 180. Returning to white line.")

        elif self.state == self.STATE_BOX_RETURN_TO_LINE:
            resume_turn_direction = self.side_sign(self.active_box_side)

            self.configure_line_cross_sequence(
                speed=self.box_return_speed,
                turn_direction=resume_turn_direction,
                next_state=self.STATE_FOLLOW_LINE
            )
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info(
                f"Returning from {self.active_box_side} box. Reusing line-cross sequence to resume path."
            )

        elif self.state == self.STATE_TASK2_DONE:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

        if self.print_distances_every_frame and self.measurement_started:
            self.get_logger().info(
                f"STATE={self.state} "
                f"left_raw={self.fmt_range(self.left_range_raw)} left_f={self.fmt_range(self.left_range)} "
                f"right_raw={self.fmt_range(self.right_range_raw)} right_f={self.fmt_range(self.right_range)} "
                f"front_raw={self.fmt_range(self.front_range_raw)} front_f={self.fmt_range(self.front_range)} "
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

        cv2.putText(
            vis,
            f"LEFT count={self.left_box_count} RIGHT count={self.right_box_count}",
            (10, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2
        )

        cv2.putText(
            vis,
            f"boxes_completed={self.boxes_completed}/{self.target_box_count}",
            (10, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 220, 120),
            2
        )

        cv2.putText(
            vis,
            f"finish_counter={self.finish_wall_counter}/{self.task2_finish_wall_frames}",
            (10, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (120, 255, 220),
            2
        )

        cv2.putText(
            vis,
            f"active_box={self.active_box_side} red_found={red_found}",
            (10, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 255, 200),
            2
        )

        cv2.putText(
            vis,
            f"pick='{self.pick_feedback_text}'",
            (10, 275),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 100),
            2
        )

        cv2.putText(
            vis,
            f"cmd v={twist.linear.x:.2f} w={twist.angular.z:.2f}",
            (10, 305),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.imshow("camera", vis)
        cv2.imshow("mask", mask)
        cv2.imshow("bottom_mask", bottom_mask)
        cv2.imshow("red_mask", red_mask)

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
                f"main_area={int(area)} bottom_area={int(bottom_area)} red_area={int(red_area)} "
                f"left={self.display_range(self.left_range)} "
                f"right={self.display_range(self.right_range)} "
                f"front={self.display_range(self.front_range)} "
                f"counts(L,R)=({self.left_box_count},{self.right_box_count}) "
                f"boxes_completed={self.boxes_completed}/{self.target_box_count} "
                f"finish_counter={self.finish_wall_counter}/{self.task2_finish_wall_frames} "
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
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()