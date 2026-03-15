#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, Range
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import cv2
import numpy as np
import matplotlib.pyplot as plt


class WhiteLineFollowerWithTurnAndBoxCount(Node):
    def __init__(self):
        super().__init__('task2')

        # =========================
        # Topics
        # =========================
        self.declare_parameter('image_topic', '/camera/image/image_color')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('left_range_topic', '/robocop/ds_left')
        self.declare_parameter('right_range_topic', '/robocop/ds_right')

        # =========================
        # Motion control
        # =========================
        self.declare_parameter('forward_speed', 0.12)
        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('kp', 0.004)
        self.declare_parameter('max_angular', 1.2)

        self.declare_parameter('extra_forward_distance', 0.18)

        self.declare_parameter('turn_left_angular_speed', 0.8)
        self.declare_parameter('turn_left_90_time', 2.0)

        # new
        self.declare_parameter('post_turn_wait_time', 5.0)

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
        # Distance sensor / box counting
        # =========================
        self.declare_parameter('box_detect_distance', 0.5)
        self.declare_parameter('box_release_distance', 0.15)
        self.declare_parameter('box_detect_frames', 10)
        self.declare_parameter('box_release_frames', 10)
        self.declare_parameter('print_distances_every_frame', False)

        # =========================
        # Read params
        # =========================
        self.image_topic = self.get_parameter('image_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.left_range_topic = self.get_parameter('left_range_topic').value
        self.right_range_topic = self.get_parameter('right_range_topic').value

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

        self.box_detect_distance = float(self.get_parameter('box_detect_distance').value)
        self.box_release_distance = float(self.get_parameter('box_release_distance').value)
        self.box_detect_frames = int(self.get_parameter('box_detect_frames').value)
        self.box_release_frames = int(self.get_parameter('box_release_frames').value)
        self.print_distances_every_frame = bool(
            self.get_parameter('print_distances_every_frame').value
        )

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

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # =========================
        # State machine
        # =========================
        self.STATE_APPROACH_LINE = 'APPROACH_LINE'
        self.STATE_WAIT_LINE_DISAPPEAR = 'WAIT_LINE_DISAPPEAR'
        self.STATE_VERIFY_ON_LINE_FORWARD = 'VERIFY_ON_LINE_FORWARD'
        self.STATE_TURN_LEFT_90 = 'TURN_LEFT_90'
        self.STATE_POST_TURN_WAIT = 'POST_TURN_WAIT'
        self.STATE_FOLLOW_LINE = 'FOLLOW_LINE'

        self.state = self.STATE_APPROACH_LINE
        self.turn_start_time = None
        self.extra_forward_start_time = None
        self.post_turn_wait_start_time = None

        self.bottom_line_seen = False
        self.bottom_gone_counter = 0

        self.left_range = math.inf
        self.right_range = math.inf

        self.measurement_started = False

        self.left_box_active = False
        self.right_box_active = False
        self.left_box_count = 0
        self.right_box_count = 0

        self.left_detect_counter = 0
        self.left_release_counter = 0
        self.right_detect_counter = 0
        self.right_release_counter = 0

        self.history_t = []
        self.history_left = []
        self.history_right = []
        self.measurement_start_wall_time = None

        self.frame_count = 0
        self.last_log_time = self.get_clock().now()

        cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("bottom_mask", cv2.WINDOW_NORMAL)

        self.get_logger().info(f"Subscribing image: {self.image_topic}")
        self.get_logger().info(f"Publishing cmd_vel: {self.cmd_vel_topic}")
        self.get_logger().info(f"Left range topic: {self.left_range_topic}")
        self.get_logger().info(f"Right range topic: {self.right_range_topic}")
        self.get_logger().info(
            f"Sequence: forward -> pass line -> extra forward {self.extra_forward_distance:.2f} m "
            f"-> left 90 -> wait {self.post_turn_wait_time:.2f}s -> follow line -> start distance measure + box count"
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

    def valid_range(self, x: float) -> bool:
        return not math.isinf(x) and not math.isnan(x) and x > 0.0

    def display_range(self, x: float) -> str:
        if not self.measurement_started:
            return "OFF"
        return self.fmt_range(x)

    def left_range_cb(self, msg: Range):
        self.left_range = float(msg.range)

    def right_range_cb(self, msg: Range):
        self.right_range = float(msg.range)

    def build_white_mask(self, bgr_img):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_low, self.s_low, self.v_low], dtype=np.uint8)
        upper = np.array([self.h_high, self.s_high, self.v_high], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return mask

    def start_measurement(self):
        if self.measurement_started:
            return

        self.measurement_started = True
        self.measurement_start_wall_time = time.time()

        self.left_box_active = False
        self.right_box_active = False

        self.left_detect_counter = 0
        self.left_release_counter = 0
        self.right_detect_counter = 0
        self.right_release_counter = 0

        self.history_t.clear()
        self.history_left.clear()
        self.history_right.clear()

        self.get_logger().info("Started distance measurement, storage, and box counting.")

    def record_sensor_values(self):
        if not self.measurement_started or self.state != self.STATE_FOLLOW_LINE:
            return

        t = time.time() - self.measurement_start_wall_time
        self.history_t.append(t)
        self.history_left.append(self.left_range if self.valid_range(self.left_range) else np.nan)
        self.history_right.append(self.right_range if self.valid_range(self.right_range) else np.nan)

    def update_one_sensor_box_count(self, side_name, current_range,
                                    active_flag, detect_counter, release_counter, count_value):
        if not self.valid_range(current_range):
            detect_counter = 0
            release_counter = 0
            return active_flag, detect_counter, release_counter, count_value

        if not active_flag:
            if current_range < self.box_detect_distance:
                detect_counter += 1
            else:
                detect_counter = 0

            if detect_counter >= self.box_detect_frames:
                active_flag = True
                count_value += 1
                release_counter = 0
                detect_counter = 0
                self.get_logger().info(
                    f"{side_name} box detected. count={count_value}, range={self.fmt_range(current_range)} m"
                )
        else:
            if current_range > self.box_release_distance:
                release_counter += 1
            else:
                release_counter = 0

            if release_counter >= self.box_release_frames:
                active_flag = False
                release_counter = 0
                detect_counter = 0

        return active_flag, detect_counter, release_counter, count_value

    def update_box_counts(self):
        if not self.measurement_started or self.state != self.STATE_FOLLOW_LINE:
            return

        (
            self.left_box_active,
            self.left_detect_counter,
            self.left_release_counter,
            self.left_box_count
        ) = self.update_one_sensor_box_count(
            "LEFT",
            self.left_range,
            self.left_box_active,
            self.left_detect_counter,
            self.left_release_counter,
            self.left_box_count
        )

        (
            self.right_box_active,
            self.right_detect_counter,
            self.right_release_counter,
            self.right_box_count
        ) = self.update_one_sensor_box_count(
            "RIGHT",
            self.right_range,
            self.right_box_active,
            self.right_detect_counter,
            self.right_release_counter,
            self.right_box_count
        )

    def plot_sensor_history(self):
        if len(self.history_t) == 0:
            self.get_logger().info("No sensor history recorded. Skipping plot.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.history_t, self.history_left, label='Left sensor')
        plt.plot(self.history_t, self.history_right, label='Right sensor')
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (m)')
        plt.title('Left and Right Distance Sensor Values')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

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

        twist = Twist()

        if self.state == self.STATE_APPROACH_LINE:
            twist.linear.x = self.forward_speed
            twist.angular.z = 0.0

            if bottom_area > self.bottom_min_area:
                self.bottom_line_seen = True
                self.state = self.STATE_WAIT_LINE_DISAPPEAR
                self.bottom_gone_counter = 0
                self.get_logger().info("White line reached robot area. Waiting until it disappears...")

        elif self.state == self.STATE_WAIT_LINE_DISAPPEAR:
            twist.linear.x = self.forward_speed
            twist.angular.z = 0.0

            if bottom_area > self.bottom_min_area:
                self.bottom_gone_counter = 0
            else:
                self.bottom_gone_counter += 1

            if self.bottom_line_seen and self.bottom_gone_counter >= self.line_gone_frames:
                self.state = self.STATE_VERIFY_ON_LINE_FORWARD
                self.extra_forward_start_time = self.get_clock().now()
                self.get_logger().info(
                    f"White line passed under robot. Moving extra {self.extra_forward_distance:.2f} m before turn."
                )

        elif self.state == self.STATE_VERIFY_ON_LINE_FORWARD:
            twist.linear.x = self.forward_speed
            twist.angular.z = 0.0

            elapsed = (self.get_clock().now() - self.extra_forward_start_time).nanoseconds / 1e9
            if elapsed >= self.extra_forward_time:
                self.state = self.STATE_TURN_LEFT_90
                self.turn_start_time = self.get_clock().now()
                self.get_logger().info("Extra forward motion done. Starting left 90 turn.")

        elif self.state == self.STATE_TURN_LEFT_90:
            twist.linear.x = 0.0
            twist.angular.z = self.turn_left_angular_speed

            elapsed = (self.get_clock().now() - self.turn_start_time).nanoseconds / 1e9
            if elapsed >= self.turn_left_90_time:
                self.state = self.STATE_POST_TURN_WAIT
                self.post_turn_wait_start_time = self.get_clock().now()
                self.get_logger().info(
                    f"Left 90 turn complete. Waiting {self.post_turn_wait_time:.2f} s before white-line follower."
                )

        elif self.state == self.STATE_POST_TURN_WAIT:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

            elapsed = (self.get_clock().now() - self.post_turn_wait_start_time).nanoseconds / 1e9
            if elapsed >= self.post_turn_wait_time:
                self.state = self.STATE_FOLLOW_LINE
                self.start_measurement()
                self.get_logger().info("Post-turn wait complete. Starting white-line follower.")

        elif self.state == self.STATE_FOLLOW_LINE:
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

            self.update_box_counts()
            self.record_sensor_values()

        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

        if self.print_distances_every_frame and self.measurement_started and self.state == self.STATE_FOLLOW_LINE:
            self.get_logger().info(
                f"STATE={self.state} "
                f"left={self.fmt_range(self.left_range)} m "
                f"right={self.fmt_range(self.right_range)} m "
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
            cv2.circle(vis, (cx_vis, cy_vis), 8, (0, 0, 255), -1)
            cv2.line(vis, (w // 2, y0), (w // 2, h - 1), (255, 255, 0), 2)

        cv2.putText(vis, f"STATE: {self.state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(vis, f"main_area={int(area)} bottom_area={int(bottom_area)}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(
            vis,
            f"left_range={self.display_range(self.left_range)} right_range={self.display_range(self.right_range)}",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            vis,
            f"LEFT count={self.left_box_count} RIGHT count={self.right_box_count}",
            (10, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2
        )

        cv2.putText(
            vis,
            f"thr_detect={self.box_detect_distance:.2f} thr_release={self.box_release_distance:.2f}",
            (10, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 255, 200),
            2
        )

        cv2.putText(vis, f"cmd v={twist.linear.x:.2f} w={twist.angular.z:.2f}", (10, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("camera", vis)
        cv2.imshow("mask", mask)
        cv2.imshow("bottom_mask", bottom_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quit requested. Stopping robot.")
            stop = Twist()
            self.cmd_pub.publish(stop)
            self.plot_sensor_history()
            rclpy.shutdown()
            cv2.destroyAllWindows()
            return

        self.frame_count += 1
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds > 1_000_000_000:
            self.last_log_time = now
            self.get_logger().info(
                f"fps~{self.frame_count} state={self.state} "
                f"main_area={int(area)} bottom_area={int(bottom_area)} "
                f"left_range={self.display_range(self.left_range)} right_range={self.display_range(self.right_range)} "
                f"counts(L,R)=({self.left_box_count},{self.right_box_count}) "
                f"stored_samples={len(self.history_t)} "
                f"cmd(v,w)=({twist.linear.x:.2f},{twist.angular.z:.2f})"
            )
            self.frame_count = 0


def main():
    rclpy.init()
    node = WhiteLineFollowerWithTurnAndBoxCount()
    try:
        rclpy.spin(node)
    finally:
        try:
            stop = Twist()
            node.cmd_pub.publish(stop)
        except Exception:
            pass

        try:
            node.plot_sensor_history()
        except Exception as e:
            node.get_logger().error(f"Failed to plot sensor history: {e}")

        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()