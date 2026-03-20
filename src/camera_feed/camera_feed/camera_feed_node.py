import cv2

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class CameraFeedNode(Node):
    def __init__(self):
        super().__init__('camera_feed_node')

        self.bridge = CvBridge()

        self.camera_index = 0
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30.0

        # Match your old Webots topic name here
        self.image_topic = '/camera/image/image_color'

        self.image_pub = self.create_publisher(Image, self.image_topic, 10)

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.get_logger().error(f'Cannot open camera index {self.camera_index}')
            raise RuntimeError(f'Cannot open camera index {self.camera_index}')

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.publish_frame)

        self.frame_count = 0
        self.get_logger().info('Camera feed node started.')
        self.get_logger().info(f'Publishing images to: {self.image_topic}')

    def publish_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warning('Failed to grab frame from camera')
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'

        self.image_pub.publish(msg)

        self.frame_count += 1
        if self.frame_count % 60 == 0:
            self.get_logger().info(f'Published {self.frame_count} frames')

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraFeedNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()