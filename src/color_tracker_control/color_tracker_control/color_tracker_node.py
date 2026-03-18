import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image


class ColorTrackerNode(Node):
    def __init__(self):
        super().__init__('color_tracker_node')

        self.bridge = CvBridge()

        # ---------------- Topics ----------------
        self.image_topic = '/camera/image_raw'
        self.cmd_vel_topic = '/cmd_vel'

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # ---------------- Tracking settings ----------------
        # Default: red object (two HSV ranges because red wraps around HSV hue)
        self.use_red_dual_range = True

        self.lower_red_1 = np.array([0, 120, 70], dtype=np.uint8)
        self.upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)

        self.lower_red_2 = np.array([170, 120, 70], dtype=np.uint8)
        self.upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

        # Example alternative if you later want blue:
        # self.use_red_dual_range = False
        # self.lower_color = np.array([100, 120, 70], dtype=np.uint8)
        # self.upper_color = np.array([130, 255, 255], dtype=np.uint8)

        # ---------------- Motion tuning ----------------
        self.max_linear_speed = 0.12
        self.max_angular_speed = 1.2

        self.forward_speed_gain = 0.4
        self.angular_gain = 1.5

        # Desired apparent object size as fraction of image area
        self.target_area_ratio = 0.06
        self.area_tolerance = 0.015

        # Ignore tiny detections
        self.min_area_ratio = 0.002

        # If no object is seen, stop
        self.stop_when_lost = True

        # Search behavior if object is lost
        self.search_angular_speed = 0.4

        # Debug window
        self.show_debug = True

        self.get_logger().info('Color tracker node started.')
        self.get_logger().info(f'Subscribing to: {self.image_topic}')
        self.get_logger().info(f'Publishing cmd_vel to: {self.cmd_vel_topic}')

    def clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def publish_stop(self):
        msg = Twist()
        self.cmd_pub.publish(msg)

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if self.use_red_dual_range:
            mask1 = cv2.inRange(hsv, self.lower_red_1, self.upper_red_1)
            mask2 = cv2.inRange(hsv, self.lower_red_2, self.upper_red_2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, self.lower_color, self.upper_color)

        # Clean noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = frame.shape[:2]
        image_center_x = w / 2.0
        image_area = float(w * h)

        cmd = Twist()

        found = False

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            area_ratio = area / image_area

            if area_ratio >= self.min_area_ratio:
                moments = cv2.moments(largest)

                if moments['m00'] > 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])

                    found = True

                    # Horizontal error: -1 to +1 approximately
                    x_error = (image_center_x - cx) / image_center_x

                    # Angular velocity:
                    # object left  -> positive turn left
                    # object right -> negative turn right
                    angular_z = self.angular_gain * x_error
                    angular_z = self.clamp(
                        angular_z,
                        -self.max_angular_speed,
                        self.max_angular_speed
                    )

                    # Area control:
                    # too small -> move forward
                    # too large -> move backward
                    area_error = self.target_area_ratio - area_ratio

                    if abs(area_error) < self.area_tolerance:
                        linear_x = 0.0
                    else:
                        linear_x = self.forward_speed_gain * area_error
                        linear_x = self.clamp(
                            linear_x,
                            -self.max_linear_speed,
                            self.max_linear_speed
                        )

                    cmd.linear.x = linear_x
                    cmd.angular.z = angular_z

                    # Debug drawing
                    if self.show_debug:
                        cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)
                        cv2.line(frame, (int(image_center_x), 0), (int(image_center_x), h), (0, 255, 255), 2)

                        debug_text_1 = f'area_ratio={area_ratio:.4f}'
                        debug_text_2 = f'linear_x={linear_x:.3f} angular_z={angular_z:.3f}'
                        cv2.putText(frame, debug_text_1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, debug_text_2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if not found:
            if self.stop_when_lost:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = self.search_angular_speed

            if self.show_debug:
                cv2.putText(frame, 'OBJECT LOST', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        self.cmd_pub.publish(cmd)

        if self.show_debug:
            cv2.imshow('color_mask', mask)
            cv2.imshow('color_tracker_debug', frame)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ColorTrackerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        if node.show_debug:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()