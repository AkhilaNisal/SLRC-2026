#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range


class FrontStopNode(Node):
    def __init__(self):
        super().__init__('front_stop_node')

        self.declare_parameter('slow_distance_m', 0.30)    # start slowing
        self.declare_parameter('stop_distance_m', 0.20)    # full stop
        self.declare_parameter('max_speed', 0.10)          # max forward speed

        self.slow_distance_m = self.get_parameter('slow_distance_m').value
        self.stop_distance_m = self.get_parameter('stop_distance_m').value
        self.max_speed = self.get_parameter('max_speed').value

        self.front_distance = None
        self.state = "waiting"

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.dist_sub = self.create_subscription(
            Range,
            '/robocop/ds_front',
            self.distance_callback,
            10
        )

        self.timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        self.get_logger().info('Front stop node started')
        self.get_logger().info(f'Slow distance: {self.slow_distance_m:.2f} m')
        self.get_logger().info(f'Stop distance: {self.stop_distance_m:.2f} m')
        self.get_logger().info(f'Max speed: {self.max_speed:.2f} m/s')

    def distance_callback(self, msg: Range):
        self.front_distance = msg.range

    def publish_cmd(self, linear_x: float):
        msg = Twist()
        msg.linear.x = linear_x
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

    def compute_speed(self, distance: float) -> float:
        # Stop zone
        if distance <= self.stop_distance_m:
            return 0.0

        # Full speed zone
        if distance >= self.slow_distance_m:
            return self.max_speed

        # Continuous slowdown zone:
        # map distance from [stop_distance, slow_distance]
        # to speed from [0.0, max_speed]
        ratio = ((distance - self.stop_distance_m) /
                 (self.slow_distance_m - self.stop_distance_m))

        speed = ratio * self.max_speed

        # Optional minimum crawl speed to avoid too-weak motion
        if speed < 0.02:
            speed = 0.02

        return speed

    def control_loop(self):
        if self.front_distance is None:
            if self.state != "waiting":
                self.get_logger().warn('Waiting for /robocop/ds_front data...')
                self.state = "waiting"
            self.publish_cmd(0.0)
            return

        speed = self.compute_speed(self.front_distance)

        if speed == 0.0:
            if self.state != "stopped":
                self.get_logger().warn(
                    f'STOP: distance={self.front_distance:.3f} m'
                )
                self.state = "stopped"
        elif speed < self.max_speed:
            if self.state != "slowing":
                self.get_logger().info(
                    f'SLOWING: distance={self.front_distance:.3f} m'
                )
                self.state = "slowing"
        else:
            if self.state != "forward":
                self.get_logger().info(
                    f'FORWARD: distance={self.front_distance:.3f} m'
                )
                self.state = "forward"

        self.publish_cmd(speed)


def main(args=None):
    rclpy.init(args=args)
    node = FrontStopNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.publish_cmd(0.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()