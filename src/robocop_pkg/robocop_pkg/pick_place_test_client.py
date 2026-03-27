#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from robot_arm_interfaces.action import PickBox


class PickPlaceTestClient(Node):
    def __init__(self):
        super().__init__("pick_place_test_client")

        self.declare_parameter("arm_action_name", "/pick_box")
        self.declare_parameter("side", "LEFT")   # LEFT or RIGHT

        self.arm_action_name = str(self.get_parameter("arm_action_name").value)
        self.side = str(self.get_parameter("side").value).strip().upper()

        self.client = ActionClient(self, PickBox, self.arm_action_name)

        self.get_logger().info(
            f"Pick/place test client started. Waiting for action server: {self.arm_action_name}"
        )

        self.start_test()

    def start_test(self):
        if self.side not in ["LEFT", "RIGHT"]:
            self.get_logger().error(
                f"Invalid side='{self.side}'. Use LEFT or RIGHT."
            )
            rclpy.shutdown()
            return

        if not self.client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f"Action server not available: {self.arm_action_name}"
            )
            rclpy.shutdown()
            return

        goal_msg = PickBox.Goal()
        goal_msg.side = self.side

        self.get_logger().info(f"Sending pick/place goal with side={self.side} ...")
        send_goal_future = self.client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_cb
        )
        send_goal_future.add_done_callback(self.goal_response_cb)

    def feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f"[PICK feedback] step={fb.current_step} progress={fb.progress:.2f}"
        )

    def goal_response_cb(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f"Failed to send goal: {e}")
            rclpy.shutdown()
            return

        if not goal_handle.accepted:
            self.get_logger().error("Pick/place goal rejected by server.")
            rclpy.shutdown()
            return

        self.get_logger().info("Pick/place goal accepted.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_cb)

    def result_cb(self, future):
        try:
            result_wrap = future.result()
            result = result_wrap.result
        except Exception as e:
            self.get_logger().error(f"Failed to get result: {e}")
            rclpy.shutdown()
            return

        self.get_logger().info(
            f"PICK/PLACE result: success={result.success}, message='{result.message}'"
        )

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceTestClient()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()