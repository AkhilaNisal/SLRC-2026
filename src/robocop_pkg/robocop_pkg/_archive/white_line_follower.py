#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import cv2
import numpy as np


class WhiteLineFollower(Node):
    def __init__(self):
        super().__init__('white_line_follower')

        # Topics + control
        self.declare_parameter('image_topic', '/camera/image/image_color')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('kp', 0.004)           # steering gain
        self.declare_parameter('max_angular', 1.2)    # rad/s clamp

        # ROI + detection
        self.declare_parameter('roi_y_start', 0.60)
        self.declare_parameter('min_area', 5000)

        # HSV white threshold
        self.declare_parameter('h_low', 0)
        self.declare_parameter('s_low', 0)
        self.declare_parameter('v_low', 180)
        self.declare_parameter('h_high', 180)
        self.declare_parameter('s_high', 70)
        self.declare_parameter('v_high', 255)

        # Lost-line behavior
        self.declare_parameter('search_linear', 0.05)
        self.declare_parameter('search_angular', 0.35)

        # Read params
        self.image_topic = self.get_parameter('image_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.kp = float(self.get_parameter('kp').value)
        self.max_angular = float(self.get_parameter('max_angular').value)

        self.roi_y_start = float(self.get_parameter('roi_y_start').value)
        self.min_area = int(self.get_parameter('min_area').value)

        self.h_low = int(self.get_parameter('h_low').value)
        self.s_low = int(self.get_parameter('s_low').value)
        self.v_low = int(self.get_parameter('v_low').value)
        self.h_high = int(self.get_parameter('h_high').value)
        self.s_high = int(self.get_parameter('s_high').value)
        self.v_high = int(self.get_parameter('v_high').value)

        self.search_linear = float(self.get_parameter('search_linear').value)
        self.search_angular = float(self.get_parameter('search_angular').value)

        # ROS interfaces
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # Debug
        self.frame_count = 0
        self.last_log_time = self.get_clock().now()

        cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

        self.get_logger().info(f"✅ Subscribing to: {self.image_topic}")
        self.get_logger().info(f"✅ Publishing cmd_vel: {self.cmd_vel_topic}")
        self.get_logger().info("Press 'q' in the OpenCV window to quit.")

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]

        y0 = int(h * self.roi_y_start)
        roi = frame[y0:h, 0:w]

        # White detection in HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_low, self.s_low, self.v_low])
        upper = np.array([self.h_high, self.s_high, self.v_high])
        mask = cv2.inRange(hsv, lower, upper)

        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        M = cv2.moments(mask)
        area = M["m00"]

        twist = Twist()
        cx = None
        detected = False

        if area > self.min_area:
            cx = int(M["m10"] / area)  # centroid x in ROI
            error = float(cx - (w // 2))  # +ve => line is to right

            # P-controller steering
            ang = -self.kp * error
            ang = self.clamp(ang, -self.max_angular, self.max_angular)

            twist.linear.x = self.linear_speed
            twist.angular.z = ang
            detected = True
        else:
            # Line lost: slow + search turn
            twist.linear.x = self.search_linear
            twist.angular.z = self.search_angular

        self.cmd_pub.publish(twist)

        # ---------- Debug visualization ----------
        vis = frame.copy()
        cv2.rectangle(vis, (0, y0), (w - 1, h - 1), (0, 255, 0), 2)

        if cx is not None:
            cv2.circle(vis, (cx, y0 + (h - y0) // 2), 8, (0, 0, 255), -1)

        if detected:
            cv2.putText(
                vis,
                f"LINE area={int(area)} cx={cx} cmd: v={twist.linear.x:.2f} w={twist.angular.z:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                vis,
                f"NO LINE area={int(area)} searching... cmd: v={twist.linear.x:.2f} w={twist.angular.z:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow("camera", vis)
        cv2.imshow("mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quit requested (q pressed). Stopping robot.")
            stop = Twist()
            self.cmd_pub.publish(stop)
            rclpy.shutdown()
            cv2.destroyAllWindows()
            return

        # Log once per ~1 second
        self.frame_count += 1
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds > 1_000_000_000:
            self.last_log_time = now
            self.get_logger().info(
                f"fps~{self.frame_count} img={w}x{h} roi_y0={y0} area={int(area)} cx={cx} "
                f"cmd(v,w)=({twist.linear.x:.2f},{twist.angular.z:.2f})"
            )
            self.frame_count = 0


def main():
    rclpy.init()
    node = WhiteLineFollower()
    try:
        rclpy.spin(node)
    finally:
        # publish stop on exit
        try:
            stop = Twist()
            node.cmd_pub.publish(stop)
        except Exception:
            pass
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()