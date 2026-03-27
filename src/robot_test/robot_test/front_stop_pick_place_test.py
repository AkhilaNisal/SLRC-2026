#!/usr/bin/env python3

import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from robot_arm_interfaces.action import PickBox


class FrontStopPickPlaceTest(Node):
    def __init__(self):
        super().__init__('front_stop_pick_place_test')

        # Motion parameters
        self.declare_parameter('slow_distance_m', 0.20)      # start slowing
        self.declare_parameter('stop_distance_m', 0.13)      # full stop
        self.declare_parameter('max_speed', 0.10)            # max forward speed
        self.declare_parameter('control_period', 0.05)       # 20 Hz
        self.declare_parameter('stop_settle_sec', 1.0)       # wait before arm action

        # Action parameters
        self.declare_parameter('pick_side', 'LEFT')          # LEFT / RIGHT / RESTORE
        self.declare_parameter('pick_action_name', '/pick_box')
        self.declare_parameter('max_action_retries', 3)
        self.declare_parameter('retry_wait_sec', 1.0)

        self.slow_distance_m = float(self.get_parameter('slow_distance_m').value)
        self.stop_distance_m = float(self.get_parameter('stop_distance_m').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.control_period = float(self.get_parameter('control_period').value)
        self.stop_settle_sec = float(self.get_parameter('stop_settle_sec').value)

        self.pick_side = str(self.get_parameter('pick_side').value).strip().upper()
        self.pick_action_name = str(self.get_parameter('pick_action_name').value)
        self.max_action_retries = int(self.get_parameter('max_action_retries').value)
        self.retry_wait_sec = float(self.get_parameter('retry_wait_sec').value)

        self.front_distance = None
        self.state = "waiting"          # waiting, forward, slowing, stopped, picking, done, failed
        self.stop_reached_time = None
        self.action_in_progress = False
        self.action_done = False
        self.retry_count = 0
        self.retry_due_time = None

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.dist_sub = self.create_subscription(
            Range,
            '/robocop/ds_front',
            self.distance_callback,
            10
        )

        self.pick_client = ActionClient(self, PickBox, self.pick_action_name)

        self.timer = self.create_timer(self.control_period, self.control_loop)

        self.get_logger().info('Front stop pick/place test started')
        self.get_logger().info(f'Slow distance: {self.slow_distance_m:.2f} m')
        self.get_logger().info(f'Stop distance: {self.stop_distance_m:.2f} m')
        self.get_logger().info(f'Max speed: {self.max_speed:.2f} m/s')
        self.get_logger().info(f'Pick side: {self.pick_side}')
        self.get_logger().info(f'Pick action: {self.pick_action_name}')

    def distance_callback(self, msg: Range):
        self.front_distance = float(msg.range)

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
        # Same logic as your working FrontStopNode

        # Stop zone
        if distance <= self.stop_distance_m:
            return 0.0

        # Full speed zone
        if distance >= self.slow_distance_m:
            return self.max_speed

        # Continuous slowdown zone
        ratio = (
            (distance - self.stop_distance_m) /
            (self.slow_distance_m - self.stop_distance_m)
        )

        speed = ratio * self.max_speed

        # Minimum crawl speed
        if speed < 0.02:
            speed = 0.02

        return speed

    def send_pick_goal(self):
        if self.action_in_progress or self.action_done:
            return

        if not self.pick_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(f'Action server {self.pick_action_name} not available')
            self.schedule_retry()
            return

        goal_msg = PickBox.Goal()
        goal_msg.side = self.pick_side

        self.get_logger().info(
            f'Sending PickBox goal: side={self.pick_side}, attempt={self.retry_count + 1}'
        )

        self.action_in_progress = True
        future = self.pick_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f'Arm feedback: step={fb.current_step}, progress={fb.progress:.2f}'
        )

    def goal_response_callback(self, future):
        self.action_in_progress = False

        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'Failed to send PickBox goal: {exc}')
            self.schedule_retry()
            return

        if not goal_handle.accepted:
            self.get_logger().error('PickBox goal was rejected by action server')
            self.schedule_retry()
            return

        self.get_logger().info('PickBox goal accepted')
        self.action_in_progress = True
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        self.action_in_progress = False

        try:
            result_wrap = future.result()
            result = result_wrap.result
            status = result_wrap.status
        except Exception as exc:
            self.get_logger().error(f'Failed to get PickBox result: {exc}')
            self.schedule_retry()
            return

        self.get_logger().info(
            f'PickBox finished: success={result.success}, status={status}, message="{result.message}"'
        )

        if result.success:
            self.action_done = True
            self.state = 'done'
            self.publish_cmd(0.0)
            self.get_logger().info('Test completed successfully')
        else:
            self.schedule_retry()

    def schedule_retry(self):
        self.retry_count += 1

        if self.retry_count >= self.max_action_retries:
            self.get_logger().error('Max action retries reached. Stopping test.')
            self.action_done = True
            self.state = 'failed'
            self.publish_cmd(0.0)
            return

        self.retry_due_time = time.monotonic() + self.retry_wait_sec
        self.state = 'stopped'
        self.get_logger().warn(
            f'Retrying PickBox after {self.retry_wait_sec:.1f}s '
            f'({self.retry_count}/{self.max_action_retries - 1} retries used)'
        )

    def control_loop(self):
        if self.action_done:
            self.publish_cmd(0.0)
            return

        if self.front_distance is None:
            if self.state != "waiting":
                self.get_logger().warn('Waiting for /robocop/ds_front data...')
                self.state = "waiting"
            self.publish_cmd(0.0)
            return

        # Driving phase: identical stop logic to your working node
        if self.state in ["waiting", "forward", "slowing"]:
            speed = self.compute_speed(self.front_distance)

            if speed == 0.0:
                if self.state != "stopped":
                    self.get_logger().warn(
                        f'STOP: distance={self.front_distance:.3f} m'
                    )
                    self.state = "stopped"
                    self.stop_reached_time = time.monotonic()
                self.publish_cmd(0.0)

            elif speed < self.max_speed:
                if self.state != "slowing":
                    self.get_logger().info(
                        f'SLOWING: distance={self.front_distance:.3f} m'
                    )
                    self.state = "slowing"
                self.publish_cmd(speed)

            else:
                if self.state != "forward":
                    self.get_logger().info(
                        f'FORWARD: distance={self.front_distance:.3f} m'
                    )
                    self.state = "forward"
                self.publish_cmd(speed)

            return

        # Fully stopped, keep publishing zero
        if self.state == "stopped":
            self.publish_cmd(0.0)

            # retry case
            if self.retry_due_time is not None:
                if time.monotonic() >= self.retry_due_time:
                    self.retry_due_time = None
                    self.state = "picking"
                    self.send_pick_goal()
                return

            # first action trigger after settling
            if self.stop_reached_time is not None:
                elapsed = time.monotonic() - self.stop_reached_time
                if elapsed >= self.stop_settle_sec:
                    self.state = "picking"
                    self.send_pick_goal()
            return

        # While action is running, keep robot stopped
        if self.state == "picking":
            self.publish_cmd(0.0)
            return

        if self.state in ["done", "failed"]:
            self.publish_cmd(0.0)
            return


def main(args=None):
    rclpy.init(args=args)
    node = FrontStopPickPlaceTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()