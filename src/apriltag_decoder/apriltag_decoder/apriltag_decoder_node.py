import json
from typing import Dict, Iterable, Set, Tuple

import cv2
import rclpy
from cv_bridge import CvBridge
from pupil_apriltags import Detector
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


class AprilTagDecoderNode(Node):
    def __init__(self) -> None:
        super().__init__('apriltag_decoder_node')

        self.declare_parameter('image_topic', '/camera/image/image_color')
        self.declare_parameter('decoded_topic', '/apriltag/decoded')
        self.declare_parameter('debug_image_topic', '/apriltag/debug_image')
        self.declare_parameter('families', 'tagStandard52h13')
        self.declare_parameter('required_unique_tags', 14)
        self.declare_parameter('publish_on_each_detection', True)
        self.declare_parameter('publish_debug_image', True)

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.decoded_topic = self.get_parameter('decoded_topic').get_parameter_value().string_value
        self.debug_image_topic = self.get_parameter('debug_image_topic').get_parameter_value().string_value
        self.families = self.get_parameter('families').get_parameter_value().string_value
        self.required_unique_tags = int(
            self.get_parameter('required_unique_tags').get_parameter_value().integer_value
        )
        self.publish_on_each_detection = bool(
            self.get_parameter('publish_on_each_detection').get_parameter_value().bool_value
        )
        self.publish_debug_image = bool(
            self.get_parameter('publish_debug_image').get_parameter_value().bool_value
        )

        self.bridge = CvBridge()
        self.detector = Detector(
            families=self.families,
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        self.found_tags: Set[int] = set()
        self.final_result_published = False
        self.received_frame_count = 0

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.decoded_pub = self.create_publisher(String, self.decoded_topic, 10)
        self.debug_image_pub = self.create_publisher(
            Image,
            self.debug_image_topic,
            qos_profile_sensor_data,
        )
        self.health_timer = self.create_timer(5.0, self._health_check)

        self.get_logger().info('AprilTag decoder node started.')
        self.get_logger().info(f'Subscribing to image topic: {self.image_topic}')
        self.get_logger().info(f'Publishing decoded output topic: {self.decoded_topic}')
        if self.publish_debug_image:
            self.get_logger().info(f'Publishing debug image topic: {self.debug_image_topic}')
        self.get_logger().info(
            'If no frames arrive, try: --ros-args -p image_topic:=/camera/image_raw'
        )

    def _health_check(self) -> None:
        if self.received_frame_count == 0:
            self.get_logger().warn(
                f'No image frames received on {self.image_topic}. '
                'Check topic name and publisher QoS.'
            )

    def image_callback(self, msg: Image) -> None:
        self.received_frame_count += 1
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'cv_bridge conversion failed: {exc}')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)

        if self.publish_debug_image and detections:
            for tag in detections:
                corners = tag.corners.reshape((-1, 1, 2)).astype(int)
                cv2.polylines(frame, [corners], True, (0, 0, 255), 4)
                center = (int(tag.center[0]), int(tag.center[1]))
                cv2.circle(frame, center, 6, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f'ID:{int(tag.tag_id)}',
                    (center[0] + 8, center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                )

        new_tags = []
        for tag in detections:
            tag_id = int(tag.tag_id)
            if tag_id not in self.found_tags:
                self.found_tags.add(tag_id)
                new_tags.append(tag_id)

        if new_tags:
            self.get_logger().info(
                f'New tags: {sorted(new_tags)} | total unique={len(self.found_tags)}'
            )

            if self.publish_on_each_detection:
                self.publish_decoded_output(final=False)

        if (not self.final_result_published) and len(self.found_tags) >= self.required_unique_tags:
            self.final_result_published = True
            self.publish_decoded_output(final=True)
            self.get_logger().info('Required unique tags reached. Final decoded output published.')

        if self.publish_debug_image:
            debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)

    def publish_decoded_output(self, final: bool) -> None:
        decoded = decode_all_tags(self.found_tags)
        payload = {
            'final': final,
            'unique_count': len(self.found_tags),
            'required_unique_tags': self.required_unique_tags,
            'tag_ids': sorted(self.found_tags),
            'decoded_by_order': decoded,
        }

        message = String()
        message.data = json.dumps(payload)
        self.decoded_pub.publish(message)


def decode_key0(tag_id: int) -> Tuple[int, int, int]:
    s = str(tag_id).zfill(5)
    payload = s[1:]
    p_rev = int(payload[::-1])

    a_val = ((p_rev * 7) + 6180) % 10000
    order = a_val // 625 + 1
    remainder = a_val % 625
    x = remainder // 25
    y = remainder % 25
    return order, x, y


def decode_key1(tag_id: int) -> Tuple[int, int, int]:
    s = str(tag_id).zfill(5)
    payload = s[1:]
    p_swap = int(payload[2:] + payload[:2])

    a_val = ((p_swap * 3) + 3141) % 8750
    order = a_val // 625 + 1
    remainder = a_val % 625
    x = remainder // 25
    y = remainder % 25
    return order, x, y


def decode_key2(tag_id: int) -> Tuple[int, int, int]:
    s = str(tag_id).zfill(5)
    payload = int(s[1:])
    p_comp = 9999 - payload

    a_val = ((p_comp * 9) + 2718) % 8750
    order = a_val // 625 + 1
    remainder = a_val % 625
    x = remainder // 25
    y = remainder % 25
    return order, x, y


def decode_key3(tag_id: int) -> Tuple[int, int, int]:
    s = str(tag_id).zfill(5)
    payload = s[1:]
    p_int = int(payload[3] + payload[1] + payload[2] + payload[0])

    a_val = ((p_int * 11) + 8080) % 8750
    order = a_val // 625 + 1
    remainder = a_val % 625
    x = remainder // 25
    y = remainder % 25
    return order, x, y


def decode_key4(tag_id: int) -> Tuple[int, int, int]:
    tag_str = str(tag_id).strip()

    if not tag_str.isdigit():
        raise ValueError(f'Tag must be numeric, got {tag_id}')

    tag_str = tag_str.zfill(5)
    if len(tag_str) != 5:
        raise ValueError(f'Tag must be a 5-digit integer/string like 42305, got {tag_id}')

    payload = int(tag_str[1:])
    gray_code = payload ^ (payload // 2)
    a_val = gray_code ^ 4040

    order = (a_val // 625) + 1
    remainder = a_val % 625
    x = remainder // 25
    y = remainder % 25
    return order, x, y


def decode_single_tag(tag_id: int) -> Dict[str, int]:
    s = str(tag_id).zfill(5)
    key = int(s[0])

    if key == 0:
        order, x, y = decode_key0(tag_id)
    elif key == 1:
        order, x, y = decode_key1(tag_id)
    elif key == 2:
        order, x, y = decode_key2(tag_id)
    elif key == 3:
        order, x, y = decode_key3(tag_id)
    elif key == 4:
        order, x, y = decode_key4(tag_id)
    else:
        raise ValueError(f'Unsupported key: {key} for tag ID {tag_id}')

    return {'order': order, 'x': x, 'y': y, 'tag_id': tag_id, 'key': key}


def decode_all_tags(tag_ids: Iterable[int]) -> Dict[str, Dict[str, int]]:
    decoded_by_order: Dict[int, Dict[str, int]] = {}

    for tag_id in tag_ids:
        try:
            decoded = decode_single_tag(int(tag_id))
        except ValueError:
            continue

        order = int(decoded['order'])
        decoded_by_order[order] = {
            'x': int(decoded['x']),
            'y': int(decoded['y']),
            'tag_id': int(decoded['tag_id']),
            'key': int(decoded['key']),
        }

    return {
        str(order): values
        for order, values in sorted(decoded_by_order.items(), key=lambda item: item[0])
    }


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AprilTagDecoderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
