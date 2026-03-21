#!/usr/bin/env python3
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node

from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanningComponent
from moveit_configs_utils import MoveItConfigsBuilder

from robot_arm_interfaces.action import PickBox


ROBOT_CONFIG = MoveItConfigsBuilder(
    robot_name="my_arm",
    package_name="robot_arm_v2_moveit_config"
).robot_description_semantic(
    "config/my_arm.srdf",
    {"name": "my_arm"}
).to_dict()

ROBOT_CONFIG = {
    **ROBOT_CONFIG,
    "planning_scene_monitor": {
        "name": "planning_scene_monitor",
        "robot_description": "robot_description",
        "joint_state_topic": "/joint_states",
        "attached_collision_object_topic": "/moveit_cpp/planning_scene_monitor",
        "publish_planning_scene_topic": "/moveit_cpp/publish_planning_scene",
        "monitored_planning_scene_topic": "/moveit_cpp/monitored_planning_scene",
        "wait_for_initial_state_timeout": 10.0,
    },
    "planning_pipelines": {
        "pipeline_names": ["ompl"]
    },
    "plan_request_params": {
        "planning_attempts": 1,
        "planning_pipeline": "ompl",
        "max_velocity_scaling_factor": 1.0,
        "max_acceleration_scaling_factor": 1.0
    },
    "ompl": {
        "planning_plugins": ["ompl_interface/OMPLPlanner"],
        "request_adapters": [
            "default_planning_request_adapters/ResolveConstraintFrames",
            "default_planning_request_adapters/ValidateWorkspaceBounds",
            "default_planning_request_adapters/CheckStartStateBounds",
            "default_planning_request_adapters/CheckStartStateCollision",
        ],
        "response_adapters": [
            "default_planning_response_adapters/AddTimeOptimalParameterization",
            "default_planning_response_adapters/ValidateSolution",
            "default_planning_response_adapters/DisplayMotionPath",
        ],
        "start_state_max_bounds_error": 0.1
    }
}


class RobotArmActionServer(Node):
    def __init__(self):
        super().__init__("robot_arm_action_server")

        self.declare_parameter("action_name", "/pick_box")
        self.declare_parameter("startup_delay_sec", 1.0)
        self.declare_parameter("step_pause_sec", 0.5)
        self.declare_parameter("max_box_count", 6)

        self.declare_parameter("arm_group", "robot_arm")
        self.declare_parameter("gripper_group", "gripper")

        self.action_name = str(self.get_parameter("action_name").value)
        self.startup_delay_sec = float(self.get_parameter("startup_delay_sec").value)
        self.step_pause_sec = float(self.get_parameter("step_pause_sec").value)
        self.max_box_count = int(self.get_parameter("max_box_count").value)

        self.arm_group = str(self.get_parameter("arm_group").value)
        self.gripper_group = str(self.get_parameter("gripper_group").value)

        self.arm_joint_names = [
            "base_rotating_waste_joint",
            "rotating_waste_arm1_joint",
            "arm1_arm2_joint",
            "arm2_gripper_base_joint",
        ]

        self.gripper_joint_names = [
            "gripper_base_left_joint",
        ]

        # home = [20, 37, -72, 53]
        self.declare_parameter("home.base_rotating_waste_joint", 0.3491)
        self.declare_parameter("home.rotating_waste_arm1_joint", 0.6458)
        self.declare_parameter("home.arm1_arm2_joint", -1.2566)
        self.declare_parameter("home.arm2_gripper_base_joint", 0.9250)
        # grab = [-7, 77, 5, -80]
        self.declare_parameter("grab.base_rotating_waste_joint", -0.1222)
        self.declare_parameter("grab.rotating_waste_arm1_joint", 1.3439)
        self.declare_parameter("grab.arm1_arm2_joint", 0.0873)
        self.declare_parameter("grab.arm2_gripper_base_joint", -1.3963)

        # place1 = [-16, 12, -100, 67]
        self.declare_parameter("place1.base_rotating_waste_joint", -0.2793)
        self.declare_parameter("place1.rotating_waste_arm1_joint", 0.2094)
        self.declare_parameter("place1.arm1_arm2_joint", -1.7453)
        self.declare_parameter("place1.arm2_gripper_base_joint", 1.1694)

        # place2 = [23, -4, -92, 75]
        self.declare_parameter("place2.base_rotating_waste_joint", 0.4014)
        self.declare_parameter("place2.rotating_waste_arm1_joint", -0.0698)
        self.declare_parameter("place2.arm1_arm2_joint", -1.6057)
        self.declare_parameter("place2.arm2_gripper_base_joint", 1.3090)

        # place3 = [60, 16, -79, 81]
        self.declare_parameter("place3.base_rotating_waste_joint", 1.0472)
        self.declare_parameter("place3.rotating_waste_arm1_joint", 0.2793)
        self.declare_parameter("place3.arm1_arm2_joint", -1.3788)
        self.declare_parameter("place3.arm2_gripper_base_joint", 1.4137)

        # place4 = same as place1
        self.declare_parameter("place4.base_rotating_waste_joint", -0.2793)
        self.declare_parameter("place4.rotating_waste_arm1_joint", 0.2094)
        self.declare_parameter("place4.arm1_arm2_joint", -1.7453)
        self.declare_parameter("place4.arm2_gripper_base_joint", 1.1694)

        # place5 = same as place2
        self.declare_parameter("place5.base_rotating_waste_joint", 0.4014)
        self.declare_parameter("place5.rotating_waste_arm1_joint", -0.0698)
        self.declare_parameter("place5.arm1_arm2_joint", -1.6057)
        self.declare_parameter("place5.arm2_gripper_base_joint", 1.3090)

        # place6 = same as place3
        self.declare_parameter("place6.base_rotating_waste_joint", 1.0472)
        self.declare_parameter("place6.rotating_waste_arm1_joint", 0.2793)
        self.declare_parameter("place6.arm1_arm2_joint", -1.3788)
        self.declare_parameter("place6.arm2_gripper_base_joint", 1.4137)

        # restore = [-5, 76, 45, 35]
        self.declare_parameter("restore.base_rotating_waste_joint", -0.0873)
        self.declare_parameter("restore.rotating_waste_arm1_joint", 1.3265)
        self.declare_parameter("restore.arm1_arm2_joint", 0.7854)
        self.declare_parameter("restore.arm2_gripper_base_joint", 0.6109)

        # From SRDF:
        # gripper_close = 0
        # gripper_open  = -1.2144
        self.declare_parameter("gripper_open.gripper_base_left_joint", -0.1)
        self.declare_parameter("gripper_close.gripper_base_left_joint", -0.9144)

        self.declare_parameter("restore_box_count", 3)

        self.restore_box_count = int(self.get_parameter("restore_box_count").value)
        self.placed_box_count = 0

        self.get_logger().info("Initializing MoveItPy...")
        self.robot = MoveItPy(node_name="moveit_py", config_dict=ROBOT_CONFIG)

        self.arm: PlanningComponent = self.robot.get_planning_component(self.arm_group)
        self.gripper: PlanningComponent = self.robot.get_planning_component(self.gripper_group)

        self._action_server = ActionServer(
            self,
            PickBox,
            self.action_name,
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.get_logger().info(f"Robot arm action server started on {self.action_name}")
        self.get_logger().info(f"Arm joints: {self.arm_joint_names}")
        self.get_logger().info(f"Gripper joints: {self.gripper_joint_names}")

        self._startup_done = False
        self._startup_timer = self.create_timer(
            self.startup_delay_sec,
            self.startup_move_home_once
        )

    def goal_callback(self, goal_request):
        self.get_logger().info(f"Received PickBox goal: side={goal_request.side}")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("Received cancel request.")
        return CancelResponse.ACCEPT

    def publish_feedback(self, goal_handle, step: str, progress: float):
        feedback = PickBox.Feedback()
        feedback.current_step = step
        feedback.progress = float(progress)
        goal_handle.publish_feedback(feedback)

    def get_arm_joint_dict(self, prefix: str) -> dict:
        values = {}
        for joint_name in self.arm_joint_names:
            values[joint_name] = float(self.get_parameter(f"{prefix}.{joint_name}").value)
        return values

    def get_gripper_joint_dict(self, prefix: str) -> dict:
        values = {}
        for joint_name in self.gripper_joint_names:
            values[joint_name] = float(self.get_parameter(f"{prefix}.{joint_name}").value)
        return values

    def move_arm_joint_values(self, joint_values: dict, label: str) -> bool:
        try:
            robot_state = RobotState(self.robot.get_robot_model())
            robot_state.joint_positions = joint_values

            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(robot_state=robot_state)

            plan_result = self.arm.plan()
            if not plan_result:
                self.get_logger().error(f"Planning failed for arm joint target '{label}'")
                return False

            self.robot.execute(plan_result.trajectory, controllers=[])
            time.sleep(self.step_pause_sec)

            joint_str = ", ".join([f"{k}={v:.4f}" for k, v in joint_values.items()])
            self.get_logger().info(f"Executed arm target '{label}': {joint_str}")
            return True

        except Exception as exc:
            self.get_logger().error(f"Exception in move_arm_joint_values('{label}'): {exc}")
            return False

    def move_gripper_joint_values(self, joint_values: dict, label: str) -> bool:
        try:
            robot_state = RobotState(self.robot.get_robot_model())
            robot_state.joint_positions = joint_values

            self.gripper.set_start_state_to_current_state()
            self.gripper.set_goal_state(robot_state=robot_state)

            plan_result = self.gripper.plan()
            if not plan_result:
                self.get_logger().error(f"Planning failed for gripper joint target '{label}'")
                return False

            self.robot.execute(plan_result.trajectory, controllers=[])
            time.sleep(self.step_pause_sec)

            joint_str = ", ".join([f"{k}={v:.4f}" for k, v in joint_values.items()])
            self.get_logger().info(f"Executed gripper target '{label}': {joint_str}")
            return True

        except Exception as exc:
            self.get_logger().error(f"Exception in move_gripper_joint_values('{label}'): {exc}")
            return False

    def open_gripper(self) -> bool:
        return self.move_gripper_joint_values(
            self.get_gripper_joint_dict("gripper_open"),
            "gripper_open"
        )

    def close_gripper(self) -> bool:
        return self.move_gripper_joint_values(
            self.get_gripper_joint_dict("gripper_close"),
            "gripper_close"
        )

    def fail_result(self, goal_handle, message: str):
        self.get_logger().error(message)
        goal_handle.abort()
        result = PickBox.Result()
        result.success = False
        result.message = message
        return result

    def success_result(self, goal_handle, message: str):
        self.get_logger().info(message)
        goal_handle.succeed()
        result = PickBox.Result()
        result.success = True
        result.message = message
        return result

    def canceled_result(self, goal_handle, message: str):
        self.get_logger().warn(message)
        goal_handle.canceled()
        result = PickBox.Result()
        result.success = False
        result.message = message
        return result

    def startup_move_home_once(self):
        if self._startup_done:
            return

        self._startup_done = True
        self._startup_timer.cancel()

        self.get_logger().info("Startup: move arm to home, gripper to open")

        home_joints = self.get_arm_joint_dict("home")

        if not self.move_arm_joint_values(home_joints, "home"):
            self.get_logger().warn("Startup home failed")

        if not self.open_gripper():
            self.get_logger().warn("Startup gripper_open failed")

    def get_current_place_prefix(self) -> str:
        pose_index = min(self.placed_box_count + 1, self.max_box_count)
        return f"place{pose_index}"

    def do_pick_place_sequence(self, goal_handle, side: str, place_prefix: str):
        home_joints = self.get_arm_joint_dict("home")
        grab_joints = self.get_arm_joint_dict("grab")
        place_joints = self.get_arm_joint_dict(place_prefix)

        self.publish_feedback(goal_handle, "gripper_open", 0.15)
        if not self.open_gripper():
            return self.fail_result(goal_handle, "Failed at step: gripper_open")
        if goal_handle.is_cancel_requested:
            return self.canceled_result(goal_handle, "Goal canceled after gripper_open")

        self.publish_feedback(goal_handle, "move_grab_pose", 0.35)
        if not self.move_arm_joint_values(grab_joints, "grab"):
            return self.fail_result(goal_handle, "Failed at step: grab")
        if goal_handle.is_cancel_requested:
            return self.canceled_result(goal_handle, "Goal canceled after grab")

        self.publish_feedback(goal_handle, "gripper_close", 0.55)
        if not self.close_gripper():
            return self.fail_result(goal_handle, "Failed at step: gripper_close")
        if goal_handle.is_cancel_requested:
            return self.canceled_result(goal_handle, "Goal canceled after gripper_close")

        self.publish_feedback(goal_handle, "move_place_pose", 0.75)
        if not self.move_arm_joint_values(place_joints, place_prefix):
            return self.fail_result(goal_handle, f"Failed at step: {place_prefix}")
        if goal_handle.is_cancel_requested:
            return self.canceled_result(goal_handle, "Goal canceled after place pose")

        self.publish_feedback(goal_handle, "release_box", 0.90)
        if not self.open_gripper():
            return self.fail_result(goal_handle, "Failed at step: release gripper_open")
        if goal_handle.is_cancel_requested:
            return self.canceled_result(goal_handle, "Goal canceled after release")

        self.publish_feedback(goal_handle, "return_home", 1.00)
        if not self.move_arm_joint_values(home_joints, "home"):
            return self.fail_result(goal_handle, "Failed at step: home")

        if self.placed_box_count < self.max_box_count:
            self.placed_box_count += 1

        return self.success_result(
            goal_handle,
            f"Pick and place completed successfully for side={side}, "
            f"placed_box_count={self.placed_box_count}"
        )

    def do_restore_sequence(self, goal_handle):
        home_joints = self.get_arm_joint_dict("home")
        restore_joints = self.get_arm_joint_dict("restore")

        restore_count = max(1, min(self.restore_box_count, self.max_box_count))

        self.get_logger().info(
            f"Starting restore sequence for {restore_count} boxes "
            f"using place1..place{restore_count} -> restore"
        )

        total_major_steps = restore_count * 5 + 1
        major_step = 0

        for i in range(1, restore_count + 1):
            place_prefix = f"place{i}"
            place_joints = self.get_arm_joint_dict(place_prefix)

            if goal_handle.is_cancel_requested:
                return self.canceled_result(goal_handle, f"Restore canceled before {place_prefix}")

            major_step += 1
            self.publish_feedback(
                goal_handle,
                f"{place_prefix}_open_before_pick",
                major_step / total_major_steps
            )
            if not self.open_gripper():
                return self.fail_result(
                    goal_handle,
                    f"Restore failed: gripper_open before {place_prefix}"
                )

            if goal_handle.is_cancel_requested:
                return self.canceled_result(goal_handle, f"Restore canceled before move to {place_prefix}")

            major_step += 1
            self.publish_feedback(
                goal_handle,
                f"move_{place_prefix}",
                major_step / total_major_steps
            )
            if not self.move_arm_joint_values(place_joints, place_prefix):
                return self.fail_result(goal_handle, f"Restore failed: move to {place_prefix}")

            if goal_handle.is_cancel_requested:
                return self.canceled_result(goal_handle, f"Restore canceled at {place_prefix}")

            major_step += 1
            self.publish_feedback(
                goal_handle,
                f"close_at_{place_prefix}",
                major_step / total_major_steps
            )
            if not self.close_gripper():
                return self.fail_result(
                    goal_handle,
                    f"Restore failed: gripper_close at {place_prefix}"
                )

            if goal_handle.is_cancel_requested:
                return self.canceled_result(goal_handle, f"Restore canceled after closing at {place_prefix}")

            major_step += 1
            self.publish_feedback(
                goal_handle,
                f"move_restore_from_{place_prefix}",
                major_step / total_major_steps
            )
            if not self.move_arm_joint_values(restore_joints, "restore"):
                return self.fail_result(
                    goal_handle,
                    f"Restore failed: move to restore from {place_prefix}"
                )

            if goal_handle.is_cancel_requested:
                return self.canceled_result(goal_handle, f"Restore canceled at restore from {place_prefix}")

            major_step += 1
            self.publish_feedback(
                goal_handle,
                f"open_at_restore_from_{place_prefix}",
                major_step / total_major_steps
            )
            if not self.open_gripper():
                return self.fail_result(
                    goal_handle,
                    f"Restore failed: gripper_open at restore from {place_prefix}"
                )

        major_step += 1
        self.publish_feedback(goal_handle, "restore_return_home", major_step / total_major_steps)
        if not self.move_arm_joint_values(home_joints, "home"):
            return self.fail_result(goal_handle, "Restore failed: return home")

        return self.success_result(
            goal_handle,
            f"Restore sequence completed successfully for {restore_count} boxes."
        )

    def execute_callback(self, goal_handle):
        side = goal_handle.request.side.strip().upper()

        if side == "RESTORE":
            return self.do_restore_sequence(goal_handle)

        if side not in ["LEFT", "RIGHT"]:
            return self.fail_result(goal_handle, f"Invalid side '{goal_handle.request.side}'")

        place_prefix = self.get_current_place_prefix()

        self.get_logger().info(
            f"Executing pick-place sequence for side={side}, "
            f"target_place={place_prefix}, placed_box_count={self.placed_box_count}"
        )

        if goal_handle.is_cancel_requested:
            return self.canceled_result(goal_handle, "Goal canceled before execution")

        return self.do_pick_place_sequence(goal_handle, side, place_prefix)


def main(args=None):
    rclpy.init(args=args)
    node = RobotArmActionServer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()