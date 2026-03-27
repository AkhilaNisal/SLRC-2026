#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import cv2

from robocop_pkg.line_detection_utils import build_white_mask, line_centroid, steering_command


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

        # Debug visualization (disable on headless robot)
        self.declare_parameter('debug', False)

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
        self.debug = bool(self.get_parameter('debug').value)

        # ROS interfaces
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # Debug
        self.frame_count = 0
        self.last_log_time = self.get_clock().now()

        if self.debug:
            cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
            cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
            self.get_logger().info("Press 'q' in the OpenCV window to quit.")
        else:
            self.get_logger().info("Debug windows disabled (headless mode). Set debug:=true to enable.")

        self.get_logger().info(f"Subscribing to: {self.image_topic}")
        self.get_logger().info(f"Publishing cmd_vel: {self.cmd_vel_topic}")

    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]

        y0 = int(h * self.roi_y_start)
        roi = frame[y0:h, 0:w]

        mask = build_white_mask(roi, self.h_low, self.s_low, self.v_low,
                                self.h_high, self.s_high, self.v_high)
        cx, area = line_centroid(mask, self.min_area)
        lin, ang = steering_command(cx, w, self.kp, self.max_angular,
                                    self.linear_speed, self.search_linear, self.search_angular)

        twist = Twist()
        twist.linear.x = lin
        twist.angular.z = ang
        detected = cx is not None

        self.cmd_pub.publish(twist)

        # ---------- Debug visualization ----------
        if self.debug:
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