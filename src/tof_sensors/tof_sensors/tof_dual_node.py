#!/usr/bin/env python3

import time

import board
import busio
import adafruit_vl53l0x
from digitalio import DigitalInOut

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range


class DualTofNode(Node):
    def __init__(self):
        super().__init__('dual_tof_node')

        self.declare_parameter('left_range_topic', '/robocop/ds_left')
        self.declare_parameter('right_range_topic', '/robocop/ds_right')

        self.declare_parameter('left_frame_id', 'tof_left')
        self.declare_parameter('right_frame_id', 'tof_right')

        self.declare_parameter('publish_rate_hz', 10.0)

        self.declare_parameter('left_xshut_pin', 'D17')
        self.declare_parameter('right_xshut_pin', 'D27')

        self.declare_parameter('left_i2c_address', 0x30)
        self.declare_parameter('right_i2c_address', 0x29)

        left_topic = self.get_parameter('left_range_topic').value
        right_topic = self.get_parameter('right_range_topic').value

        self.left_frame_id = self.get_parameter('left_frame_id').value
        self.right_frame_id = self.get_parameter('right_frame_id').value

        publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)

        left_xshut_name = self.get_parameter('left_xshut_pin').value
        right_xshut_name = self.get_parameter('right_xshut_pin').value

        self.left_addr = int(self.get_parameter('left_i2c_address').value)
        self.right_addr = int(self.get_parameter('right_i2c_address').value)

        self.left_pub = self.create_publisher(Range, left_topic, 10)
        self.right_pub = self.create_publisher(Range, right_topic, 10)

        self.i2c = busio.I2C(board.SCL, board.SDA)

        self.x1 = DigitalInOut(getattr(board, left_xshut_name))
        self.x2 = DigitalInOut(getattr(board, right_xshut_name))

        self._init_sensors()

        timer_period = 1.0 / publish_rate_hz
        self.timer = self.create_timer(timer_period, self.publish_ranges)

        self.get_logger().info(
            f'Dual TOF node started. Publishing left to {left_topic}, right to {right_topic}'
        )

    def _init_sensors(self):
        self.x1.switch_to_output(value=False)
        self.x2.switch_to_output(value=False)
        time.sleep(0.5)

        self.x1.value = True
        time.sleep(0.5)
        self.sensor_left = adafruit_vl53l0x.VL53L0X(self.i2c)

        if self.left_addr != 0x29:
            self.sensor_left.set_address(self.left_addr)
            time.sleep(0.2)

        self.x2.value = True
        time.sleep(0.5)
        self.sensor_right = adafruit_vl53l0x.VL53L0X(self.i2c, address=self.right_addr)

    def _make_range_msg(self, distance_mm: int, frame_id: str) -> Range:
        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        msg.radiation_type = Range.INFRARED
        msg.field_of_view = 0.436  # about 25 degrees in radians
        msg.min_range = 0.03       # 30 mm
        msg.max_range = 2.0        # 2000 mm

        msg.range = float(distance_mm) / 1000.0
        return msg

    def publish_ranges(self):
        try:
            left_mm = self.sensor_left.range
            right_mm = self.sensor_right.range

            left_msg = self._make_range_msg(left_mm, self.left_frame_id)
            right_msg = self._make_range_msg(right_mm, self.right_frame_id)

            self.left_pub.publish(left_msg)
            self.right_pub.publish(right_msg)

        except Exception as e:
            self.get_logger().error(f'Failed to read TOF sensors: {e}')

    def destroy_node(self):
        try:
            self.x1.deinit()
            self.x2.deinit()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DualTofNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()