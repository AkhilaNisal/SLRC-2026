#!/usr/bin/env python3
import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray


class PerspectiveRectifier(Node):
    def __init__(self):
        super().__init__('perspective_rectifier')

        # Topics
        self.declare_parameter('input_image_topic', '/camera/image/image_color')
        self.declare_parameter('output_image_topic', '/camera/image_rect')
        self.declare_parameter('homography_topic', '/camera/homography')

        # Output size
        self.declare_parameter('output_width', 640)
        self.declare_parameter('output_height', 480)

        # Source points in RAW image
        # You MUST tune these for your camera.
        # Order: top-left, top-right, bottom-right, bottom-left
        self.declare_parameter('src_tl_x', 140.0)
        self.declare_parameter('src_tl_y', 170.0)

        self.declare_parameter('src_tr_x', 500.0)
        self.declare_parameter('src_tr_y', 170.0)

        self.declare_parameter('src_br_x', 620.0)
        self.declare_parameter('src_br_y', 430.0)

        self.declare_parameter('src_bl_x', 20.0)
        self.declare_parameter('src_bl_y', 430.0)

        self.input_image_topic = self.get_parameter('input_image_topic').value
        self.output_image_topic = self.get_parameter('output_image_topic').value
        self.homography_topic = self.get_parameter('homography_topic').value

        self.output_width = int(self.get_parameter('output_width').value)
        self.output_height = int(self.get_parameter('output_height').value)

        self.src_pts = np.array([
            [float(self.get_parameter('src_tl_x').value), float(self.get_parameter('src_tl_y').value)],
            [float(self.get_parameter('src_tr_x').value), float(self.get_parameter('src_tr_y').value)],
            [float(self.get_parameter('src_br_x').value), float(self.get_parameter('src_br_y').value)],
            [float(self.get_parameter('src_bl_x').value), float(self.get_parameter('src_bl_y').value)],
        ], dtype=np.float32)

        self.dst_pts = np.array([
            [0.0, 0.0],
            [self.output_width - 1.0, 0.0],
            [self.output_width - 1.0, self.output_height - 1.0],
            [0.0, self.output_height - 1.0],
        ], dtype=np.float32)

        self.H = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)

        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image,
            self.input_image_topic,
            self.image_cb,
            10
        )
        self.rect_pub = self.create_publisher(Image, self.output_image_topic, 10)
        self.H_pub = self.create_publisher(Float64MultiArray, self.homography_topic, 10)

        cv2.namedWindow("raw_with_src_points", cv2.WINDOW_NORMAL)
        cv2.namedWindow("rectified_top_view", cv2.WINDOW_NORMAL)

        self.get_logger().info(f"Input image: {self.input_image_topic}")
        self.get_logger().info(f"Rectified output: {self.output_image_topic}")
        self.get_logger().info(f"Homography topic: {self.homography_topic}")

    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        rect = cv2.warpPerspective(frame, self.H, (self.output_width, self.output_height))

        # Publish rectified image
        rect_msg = self.bridge.cv2_to_imgmsg(rect, encoding='bgr8')
        rect_msg.header = msg.header
        self.rect_pub.publish(rect_msg)

        # Publish homography matrix
        H_msg = Float64MultiArray()
        H_msg.data = self.H.flatten().tolist()
        self.H_pub.publish(H_msg)

        # Debug raw view with selected source polygon
        raw_vis = frame.copy()
        pts = self.src_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(raw_vis, [pts], True, (0, 255, 0), 2)

        for i, p in enumerate(self.src_pts.astype(np.int32)):
            cv2.circle(raw_vis, tuple(p), 6, (0, 0, 255), -1)
            cv2.putText(raw_vis, f"S{i}", (p[0] + 5, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("raw_with_src_points", raw_vis)
        cv2.imshow("rectified_top_view", rect)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rclpy.shutdown()

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = PerspectiveRectifier()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()



