#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class Task3Node(Node):
    def __init__(self):
        super().__init__('task3')

        # =========================
        # Topics
        # =========================
        self.declare_parameter('image_topic', '/camera/image/image_color')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        # =========================
        # Motion
        # =========================
        self.declare_parameter('linear_speed', 0.14)
        self.declare_parameter('search_linear', 0.04)
        self.declare_parameter('search_angular', 0.30)

        self.declare_parameter('kp_line', 0.004)
        self.declare_parameter('max_angular_line', 1.2)

        self.declare_parameter('turn_left_angular_speed', 0.8)
        self.declare_parameter('turn_left_90_time', 2.15)
        self.declare_parameter('post_turn_wait_time', 0.8)

        self.declare_parameter('initial_forward_distance', 0.50)
        self.declare_parameter('initial_forward_speed', 0.12)

        self.declare_parameter('junction_turn_speed', 0.75)
        self.declare_parameter('junction_turn_90_time', 2.05)
        self.declare_parameter('junction_confirm_frames', 4)

        self.declare_parameter('pre_turn_distance', 0.3)
        self.declare_parameter('pre_turn_speed', 0.10)

        # =========================
        # White line detection
        # =========================
        self.declare_parameter('roi_y_start', 0.9)
        self.declare_parameter('min_area', 5000)

        self.declare_parameter('h_low', 0)
        self.declare_parameter('s_low', 0)
        self.declare_parameter('v_low', 180)
        self.declare_parameter('h_high', 180)
        self.declare_parameter('s_high', 70)
        self.declare_parameter('v_high', 255)

        # =========================
        # Branch detection
        # =========================
        self.declare_parameter('branch_side', 'LEFT')   # LEFT or RIGHT
        self.declare_parameter('side_roi_x_ratio', 0.34)
        self.declare_parameter('bottom_strip_height_ratio', 0.15)
        self.declare_parameter('bottom_branch_min_area', 2000)

        # =========================
        # Read params
        # =========================
        self.image_topic = self.get_parameter('image_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.search_linear = float(self.get_parameter('search_linear').value)
        self.search_angular = float(self.get_parameter('search_angular').value)

        self.kp_line = float(self.get_parameter('kp_line').value)
        self.max_angular_line = float(self.get_parameter('max_angular_line').value)

        self.turn_left_angular_speed = float(self.get_parameter('turn_left_angular_speed').value)
        self.turn_left_90_time = float(self.get_parameter('turn_left_90_time').value)
        self.post_turn_wait_time = float(self.get_parameter('post_turn_wait_time').value)

        self.initial_forward_distance = float(self.get_parameter('initial_forward_distance').value)
        self.initial_forward_speed = float(self.get_parameter('initial_forward_speed').value)
        self.initial_forward_time = (
            self.initial_forward_distance / self.initial_forward_speed
            if self.initial_forward_speed > 1e-6 else 0.0
        )

        self.junction_turn_speed = float(self.get_parameter('junction_turn_speed').value)
        self.junction_turn_90_time = float(self.get_parameter('junction_turn_90_time').value)
        self.junction_confirm_frames = int(self.get_parameter('junction_confirm_frames').value)

        self.pre_turn_distance = float(self.get_parameter('pre_turn_distance').value)
        self.pre_turn_speed = float(self.get_parameter('pre_turn_speed').value)
        self.pre_turn_time = (
            self.pre_turn_distance / self.pre_turn_speed
            if self.pre_turn_speed > 1e-6 else 0.0
        )

        self.roi_y_start = float(self.get_parameter('roi_y_start').value)
        self.min_area = int(self.get_parameter('min_area').value)

        self.h_low = int(self.get_parameter('h_low').value)
        self.s_low = int(self.get_parameter('s_low').value)
        self.v_low = int(self.get_parameter('v_low').value)
        self.h_high = int(self.get_parameter('h_high').value)
        self.s_high = int(self.get_parameter('s_high').value)
        self.v_high = int(self.get_parameter('v_high').value)

        self.branch_side = str(self.get_parameter('branch_side').value).strip().upper()
        self.side_roi_x_ratio = float(self.get_parameter('side_roi_x_ratio').value)
        self.bottom_strip_height_ratio = float(self.get_parameter('bottom_strip_height_ratio').value)
        self.bottom_branch_min_area = int(self.get_parameter('bottom_branch_min_area').value)

        if self.branch_side not in ['LEFT', 'RIGHT']:
            self.branch_side = 'LEFT'
            self.get_logger().warn("Invalid branch_side. Using LEFT.")

        # =========================
        # ROS interfaces
        # =========================
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_cb, qos_profile_sensor_data
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # =========================
        # States
        # =========================
        self.STATE_INITIAL_TURN = 'INITIAL_TURN'
        self.STATE_INITIAL_WAIT = 'INITIAL_WAIT'
        self.STATE_INITIAL_FORWARD = 'INITIAL_FORWARD'
        self.STATE_FOLLOW_LINE_TO_BRANCH = 'FOLLOW_LINE_TO_BRANCH'
        self.STATE_MOVE_FORWARD_BEFORE_TURN = 'MOVE_FORWARD_BEFORE_TURN'
        self.STATE_TURN_AT_BRANCH = 'TURN_AT_BRANCH'
        self.STATE_POST_BRANCH_WAIT = 'POST_BRANCH_WAIT'
        self.STATE_FOLLOW_LINE_AFTER_BRANCH = 'FOLLOW_LINE_AFTER_BRANCH'

        self.state = self.STATE_INITIAL_TURN
        self.turn_start_time = None
        self.wait_start_time = None
        self.forward_start_time = None
        self.pre_turn_start_time = None
        self.initial_turn_started = False

        # logic
        self.side_branch_counter = 0

        # debug
        self.frame_count = 0
        self.last_log_time = self.get_clock().now()

        cv2.namedWindow("task3_camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("task3_white_mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("task3_side_mask", cv2.WINDOW_NORMAL)

        self.get_logger().info(
            f"Task 3 started: initial left turn, forward move, line follow, detect {self.branch_side} branch at bottom region."
        )

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def build_white_mask(self, bgr_img):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_low, self.s_low, self.v_low], dtype=np.uint8)
        upper = np.array([self.h_high, self.s_high, self.v_high], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def follow_line(self, area, moments, frame_width, twist):
        if area > self.min_area:
            cx = int(moments["m10"] / area)
            error = float(cx - (frame_width // 2))
            ang = -self.kp_line * error
            ang = self.clamp(ang, -self.max_angular_line, self.max_angular_line)

            twist.linear.x = self.linear_speed
            twist.angular.z = ang
        else:
            twist.linear.x = self.search_linear
            twist.angular.z = self.search_angular

    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]

        # Main follow ROI
        y0 = int(h * self.roi_y_start)
        roi = frame[y0:h, 0:w]
        white_mask = self.build_white_mask(roi)
        M = cv2.moments(white_mask)
        area = M["m00"]

        # Bottom strip for close branch detection
        bh = int(h * self.bottom_strip_height_ratio)
        by0 = max(0, h - bh)
        bottom_roi = frame[by0:h, 0:w]

        if self.branch_side == 'LEFT':
            side_x1 = 0
            side_x2 = int(w * self.side_roi_x_ratio)
            side_bottom = bottom_roi[:, 0:side_x2]
        else:
            side_x1 = int(w * (1.0 - self.side_roi_x_ratio))
            side_x2 = w
            side_bottom = bottom_roi[:, side_x1:w]

        side_mask = self.build_white_mask(side_bottom)
        Ms = cv2.moments(side_mask)
        side_area = Ms["m00"]

        twist = Twist()

        if self.state == self.STATE_INITIAL_TURN:
            if not self.initial_turn_started:
                self.turn_start_time = self.get_clock().now()
                self.initial_turn_started = True
                self.get_logger().info("Starting initial 90-degree left turn.")

            twist.linear.x = 0.0
            twist.angular.z = self.turn_left_angular_speed

            elapsed = (self.get_clock().now() - self.turn_start_time).nanoseconds / 1e9
            if elapsed >= self.turn_left_90_time:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.state = self.STATE_INITIAL_WAIT
                self.wait_start_time = self.get_clock().now()
                self.get_logger().info("Initial left turn complete.")

        elif self.state == self.STATE_INITIAL_WAIT:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

            elapsed = (self.get_clock().now() - self.wait_start_time).nanoseconds / 1e9
            if elapsed >= self.post_turn_wait_time:
                self.state = self.STATE_INITIAL_FORWARD
                self.forward_start_time = self.get_clock().now()
                self.get_logger().info(
                    f"Moving forward {self.initial_forward_distance:.2f} m before line follow."
                )

        elif self.state == self.STATE_INITIAL_FORWARD:
            twist.linear.x = self.initial_forward_speed
            twist.angular.z = 0.0

            elapsed = (self.get_clock().now() - self.forward_start_time).nanoseconds / 1e9
            if elapsed >= self.initial_forward_time:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.state = self.STATE_FOLLOW_LINE_TO_BRANCH
                self.side_branch_counter = 0
                self.get_logger().info("Initial forward move complete. Starting line follow.")

        elif self.state == self.STATE_FOLLOW_LINE_TO_BRANCH:
            self.follow_line(area, M, w, twist)

            if side_area > self.bottom_branch_min_area:
                self.side_branch_counter += 1
            else:
                self.side_branch_counter = 0

            if self.side_branch_counter >= self.junction_confirm_frames:
                self.state = self.STATE_MOVE_FORWARD_BEFORE_TURN
                self.pre_turn_start_time = self.get_clock().now()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.get_logger().info(
                    f"{self.branch_side} branch detected at bottom region. Moving forward {self.pre_turn_distance:.2f} m before turn."
                )

        elif self.state == self.STATE_MOVE_FORWARD_BEFORE_TURN:
            twist.linear.x = self.pre_turn_speed
            twist.angular.z = 0.0

            elapsed = (self.get_clock().now() - self.pre_turn_start_time).nanoseconds / 1e9
            if elapsed >= self.pre_turn_time:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.state = self.STATE_TURN_AT_BRANCH
                self.turn_start_time = self.get_clock().now()
                self.get_logger().info("Reached junction center. Turning now.")

        elif self.state == self.STATE_TURN_AT_BRANCH:
            twist.linear.x = 0.0
            twist.angular.z = (
                self.junction_turn_speed if self.branch_side == 'LEFT'
                else -self.junction_turn_speed
            )

            elapsed = (self.get_clock().now() - self.turn_start_time).nanoseconds / 1e9
            if elapsed >= self.junction_turn_90_time:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.state = self.STATE_POST_BRANCH_WAIT
                self.wait_start_time = self.get_clock().now()
                self.get_logger().info("Branch turn complete.")

        elif self.state == self.STATE_POST_BRANCH_WAIT:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

            elapsed = (self.get_clock().now() - self.wait_start_time).nanoseconds / 1e9
            if elapsed >= self.post_turn_wait_time:
                self.state = self.STATE_FOLLOW_LINE_AFTER_BRANCH
                self.get_logger().info("Following line after branch turn.")

        elif self.state == self.STATE_FOLLOW_LINE_AFTER_BRANCH:
            self.follow_line(area, M, w, twist)

        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

        # =========================
        # Visualization
        # =========================
        vis = frame.copy()

        cv2.rectangle(vis, (0, y0), (w - 1, h - 1), (0, 255, 0), 2)
        cv2.putText(
            vis, "FOLLOW ROI", (10, max(25, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        cv2.rectangle(vis, (side_x1, by0), (side_x2 - 1, h - 1), (255, 255, 0), 2)
        cv2.putText(
            vis, f"{self.branch_side} BOTTOM BRANCH ROI", (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
        )

        if area > self.min_area:
            cx_vis = int(M["m10"] / area)
            cy_vis = y0 + (h - y0) // 2
            cv2.circle(vis, (cx_vis, cy_vis), 8, (0, 255, 255), -1)
            cv2.line(vis, (w // 2, y0), (w // 2, h - 1), (0, 0, 255), 1)

        cv2.putText(
            vis, f"STATE: {self.state}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

        cv2.putText(
            vis, f"line_area={int(area)} side_bottom_area={int(side_area)}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        cv2.putText(
            vis, f"branch_counter={self.side_branch_counter}", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 150), 2
        )

        cv2.putText(
            vis, f"cmd v={twist.linear.x:.2f} w={twist.angular.z:.2f}", (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        cv2.imshow("task3_camera", vis)
        cv2.imshow("task3_white_mask", white_mask)
        cv2.imshow("task3_side_mask", side_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quit requested.")
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
                f"line_area={int(area)} side_bottom_area={int(side_area)} "
                f"branch_counter={self.side_branch_counter} "
                f"cmd(v,w)=({twist.linear.x:.2f},{twist.angular.z:.2f})"
            )
            self.frame_count = 0


def main():
    rclpy.init()
    node = Task3Node()
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