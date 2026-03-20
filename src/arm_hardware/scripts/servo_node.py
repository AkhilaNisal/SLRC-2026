#!/usr/bin/env python3

from adafruit_servokit import ServoKit
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

kit = ServoKit(channels=16)

class ServoNode(Node):
    def __init__(self):
        super().__init__('servo_node')

        self.sub = self.create_subscription(
            JointState,
            '/joint_commands',
            self.callback,
            10)

    def callback(self, msg):
        for i, pos in enumerate(msg.position):
            deg = pos * 180.0 / math.pi

            # clamp
            deg = max(0, min(180, deg))

            kit.servo[i].angle = deg

def main():
    rclpy.init()
    node = ServoNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()