#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import cv2
import numpy as np


class RedBoxSeeker(Node):
    def __init__(self):
        super().__init__('red_box_seeker')

        # Topics + control
        self.declare_parameter('image_topic', '/camera/image/image_color')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        # Motion tuning
        self.declare_parameter('max_linear', 0.25)      # m/s
        self.declare_parameter('min_linear', 0.05)      # m/s (when target seen)
        self.declare_parameter('kp_ang', 0.0045)        # steering gain
        self.declare_parameter('max_angular', 1.5)      # rad/s clamp
        self.declare_parameter('turn_slowdown', 1.2)    # larger => slows more on turns

        # ROI (optional): ignore sky/top
        self.declare_parameter('roi_y_start', 0.0)      # 0.0 = full image; 0.5 = bottom half

        # Red detection (HSV)
        # Red wraps in HSV, so we use two ranges: [0..H1] and [H2..180]
        self.declare_parameter('h1_low', 0)
        self.declare_parameter('h1_high', 10)
        self.declare_parameter('h2_low', 170)
        self.declare_parameter('h2_high', 180)
        self.declare_parameter('s_low', 120)
        self.declare_parameter('v_low', 70)

        # Filtering + selection
        self.declare_parameter('min_area', 10)        # ignore tiny red noise
        self.declare_parameter('close_area', 45000)     # if target area bigger than this => reached
        self.declare_parameter('search_linear', 0.0)    # when target lost
        self.declare_parameter('search_angular', 0.35)  # rotate to find red

        # Read params
        self.image_topic = self.get_parameter('image_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.max_linear = float(self.get_parameter('max_linear').value)
        self.min_linear = float(self.get_parameter('min_linear').value)
        self.kp_ang = float(self.get_parameter('kp_ang').value)
        self.max_angular = float(self.get_parameter('max_angular').value)
        self.turn_slowdown = float(self.get_parameter('turn_slowdown').value)

        self.roi_y_start = float(self.get_parameter('roi_y_start').value)

        self.h1_low = int(self.get_parameter('h1_low').value)
        self.h1_high = int(self.get_parameter('h1_high').value)
        self.h2_low = int(self.get_parameter('h2_low').value)
        self.h2_high = int(self.get_parameter('h2_high').value)
        self.s_low = int(self.get_parameter('s_low').value)
        self.v_low = int(self.get_parameter('v_low').value)

        self.min_area = int(self.get_parameter('min_area').value)
        self.close_area = int(self.get_parameter('close_area').value)
        self.search_linear = float(self.get_parameter('search_linear').value)
        self.search_angular = float(self.get_parameter('search_angular').value)

        # ROS interfaces
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # Debug / smoothing
        self.prev_w = 0.0
        self.alpha_w = 0.35  # low-pass on angular vel (0=no smoothing, 1=full smoothing)

        cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("mask_red", cv2.WINDOW_NORMAL)

        self.get_logger().info(f"✅ Subscribing to: {self.image_topic}")
        self.get_logger().info(f"✅ Publishing cmd_vel: {self.cmd_vel_topic}")
        self.get_logger().info("Press 'q' in the OpenCV window to quit.")

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def build_red_mask(self, bgr_roi: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)

        lower1 = np.array([self.h1_low, self.s_low, self.v_low], dtype=np.uint8)
        upper1 = np.array([self.h1_high, 255, 255], dtype=np.uint8)

        lower2 = np.array([self.h2_low, self.s_low, self.v_low], dtype=np.uint8)
        upper2 = np.array([self.h2_high, 255, 255], dtype=np.uint8)

        m1 = cv2.inRange(hsv, lower1, upper1)
        m2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(m1, m2)

        # Clean noise
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return mask

    def pick_largest_blob(self, mask: np.ndarray):
        # Returns (area, cx, cy, bbox) in ROI coords, or None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0

        for c in contours:
            a = cv2.contourArea(c)
            if a < self.min_area:
                continue
            if a > best_area:
                x, y, w, h = cv2.boundingRect(c)
                M = cv2.moments(c)
                if M["m00"] <= 1e-6:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best_area = a
                best = (a, cx, cy, (x, y, w, h), c)

        return best

    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        H, W = frame.shape[:2]

        y0 = int(H * self.roi_y_start)
        roi = frame[y0:H, 0:W]

        mask = self.build_red_mask(roi)
        chosen = self.pick_largest_blob(mask)

        twist = Twist()
        reached = False
        detected = False

        if chosen is not None:
            area, cx, cy, (x, y, w, h), contour = chosen
            detected = True

            # "Reached" condition (big patch => close)
            if area >= self.close_area:
                reached = True
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            else:
                # Heading control: steer to center of image
                error_x = float(cx - (W / 2.0))  # + => target to right
                w_cmd = -self.kp_ang * error_x
                w_cmd = self.clamp(w_cmd, -self.max_angular, self.max_angular)

                # Smooth angular
                w_cmd = (1.0 - self.alpha_w) * w_cmd + self.alpha_w * self.prev_w
                self.prev_w = w_cmd

                # Speed control:
                # - faster when target is small (far)
                # - slower when turning hard (keeps a nice smooth curve)
                # Use area fraction as a distance proxy
                area_frac = float(area) / float(W * (H - y0) + 1e-6)
                # Map area_frac to a "go speed" (tune as needed)
                v_cmd = self.max_linear * (1.0 - self.clamp(area_frac * 8.0, 0.0, 0.9))
                v_cmd = self.clamp(v_cmd, self.min_linear, self.max_linear)

                # Slow down more when turning
                v_cmd = v_cmd / (1.0 + self.turn_slowdown * abs(w_cmd))

                twist.linear.x = float(v_cmd)
                twist.angular.z = float(w_cmd)
        else:
            # Target lost: rotate to search
            twist.linear.x = self.search_linear
            twist.angular.z = self.search_angular

        self.cmd_pub.publish(twist)

        # ---------------- Debug visualization ----------------
        vis = frame.copy()
        cv2.rectangle(vis, (0, y0), (W - 1, H - 1), (0, 255, 0), 2)

        if detected:
            # Draw bbox in full-frame coordinates
            _, cx, cy, (x, y, w, h), _ = chosen
            cv2.rectangle(vis, (x, y0 + y), (x + w, y0 + y + h), (255, 0, 0), 2)
            cv2.circle(vis, (cx, y0 + cy), 7, (0, 255, 255), -1)

            txt = f"RED area={int(chosen[0])} cx={cx} cmd(v,w)=({twist.linear.x:.2f},{twist.angular.z:.2f})"
            if reached:
                txt = "✅ REACHED RED BOX (stopping)"
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
        else:
            cv2.putText(vis, f"NO RED searching... cmd(v,w)=({twist.linear.x:.2f},{twist.angular.z:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("camera", vis)
        cv2.imshow("mask_red", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quit requested (q pressed). Stopping robot.")
            self.cmd_pub.publish(Twist())
            rclpy.shutdown()
            cv2.destroyAllWindows()
            return


def main():
    rclpy.init()
    node = RedBoxSeeker()
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