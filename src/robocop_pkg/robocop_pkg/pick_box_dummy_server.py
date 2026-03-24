#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer

from robot_arm_interfaces.action import PickBox


class PickBoxDummyServer(Node):
    def __init__(self):
        super().__init__('pick_box_dummy_server')

        self._action_server = ActionServer(
            self,
            PickBox,
            '/pick_box',
            self.execute_callback
        )

        self.get_logger().info("Dummy PickBox action server started on /pick_box")

    def execute_callback(self, goal_handle):
        side = goal_handle.request.side
        self.get_logger().info(f"Received pick goal for side={side}")

        feedback = PickBox.Feedback()

        steps = [
            ("moving_to_pregrasp", 0.2),
            ("opening_gripper", 0.4),
            ("approaching_box", 0.6),
            ("closing_gripper", 0.8),
            ("lifting_box", 1.0),
        ]

        for step_name, prog in steps:
            feedback.current_step = step_name
            feedback.progress = float(prog)
            goal_handle.publish_feedback(feedback)
            self.get_logger().info(f"Feedback: {step_name} progress={prog:.2f}")
            time.sleep(1.0)

        goal_handle.succeed()

        result = PickBox.Result()
        result.success = True
        result.message = f"Dummy pick completed successfully for {side}"
        return result


def main(args=None):
    rclpy.init(args=args)
    node = PickBoxDummyServer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()