#!/usr/bin/env python3

import time

import board
import busio
import digitalio
import adafruit_vl53l0x

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range


class TripleToFNode(Node):
    def __init__(self):
        super().__init__('tof_node')

        # Publish on the topics expected by your main node
        self.left_pub = self.create_publisher(Range, '/robocop/ds_left', 10)
        self.front_pub = self.create_publisher(Range, '/robocop/ds_front', 10)
        self.right_pub = self.create_publisher(Range, '/robocop/ds_right', 10)

        # I2C init
        self.i2c = busio.I2C(board.SCL, board.SDA)

        # XSHUT pins
        self.left_xshut = digitalio.DigitalInOut(board.D13)
        self.front_xshut = digitalio.DigitalInOut(board.D19)
        self.right_xshut = digitalio.DigitalInOut(board.D4)

        for pin in [self.left_xshut, self.front_xshut, self.right_xshut]:
            pin.direction = digitalio.Direction.OUTPUT
            pin.value = False

        time.sleep(0.2)

        # Bring sensors up one by one
        self.left_sensor = self._init_sensor(self.left_xshut, 0x30, "left")
        self.front_sensor = self._init_sensor(self.front_xshut, 0x31, "front")
        self.right_sensor = self._init_sensor(self.right_xshut, 0x32, "right")

        self.timer = self.create_timer(0.22, self.publish_ranges)  # ~4.5 Hz (accuracy mode: 200 ms budget)

        self.get_logger().info("Triple ToF node started successfully.")

    def _init_sensor(self, xshut_pin, new_address, name):
        xshut_pin.value = True
        time.sleep(0.1)

        sensor = adafruit_vl53l0x.VL53L0X(self.i2c)
        sensor.set_address(new_address)

        # Accuracy mode: 200 ms timing budget (vs default 33 ms)
        sensor.measurement_timing_budget = 200000

        self.get_logger().info(
            f"{name.capitalize()} sensor initialized at I2C address 0x{new_address:02X} "
            f"(accuracy mode, timing budget 200 ms)"
        )
        return sensor

    def _make_range_msg(self, frame_id, distance_mm):
        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        msg.radiation_type = Range.INFRARED
        msg.field_of_view = 0.44
        msg.min_range = 0.03
        msg.max_range = 2.0
        msg.range = float(distance_mm) / 1000.0  # mm -> m

        return msg

    def publish_ranges(self):
        try:
            left_mm = self.left_sensor.range
            front_mm = self.front_sensor.range
            right_mm = self.right_sensor.range

            self.left_pub.publish(self._make_range_msg("tof_left", left_mm))
            self.front_pub.publish(self._make_range_msg("tof_front", front_mm))
            self.right_pub.publish(self._make_range_msg("tof_right", right_mm))

        except Exception as e:
            self.get_logger().error(f"Error reading ToF sensors: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TripleToFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()