#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
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
        "planning_attempts": 5,
        "planning_pipeline": "ompl",
        "max_velocity_scaling_factor": 0.8,
        "max_acceleration_scaling_factor": 0.5
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
        self.declare_parameter("plan_settle_sec", 0.2)
        self.declare_parameter("retry_count", 3)
        self.declare_parameter("retry_pause_sec", 0.3)
        self.declare_parameter("max_box_count", 6)

        self.declare_parameter("arm_group", "robot_arm")
        self.declare_parameter("gripper_group", "gripper")

        # Distance move topic/settings (must match cmd_vel_stepper_node reasonably)
        self.declare_parameter("cmd_distance_topic", "/cmd_distance")
        self.declare_parameter("backward_distance_m", 0.05)           # 5 cm
        self.declare_parameter("distance_motion_extra_wait_sec", 0.3)

        # These are used only to estimate how long the /cmd_distance move will take
        self.declare_parameter("wheel_radius", 0.0325)
        self.declare_parameter("steps_per_rev", 200)
        self.declare_parameter("microsteps", 16)
        self.declare_parameter("distance_mode_sps", 800.0)

        self.action_name = str(self.get_parameter("action_name").value)
        self.startup_delay_sec = float(self.get_parameter("startup_delay_sec").value)
        self.step_pause_sec = float(self.get_parameter("step_pause_sec").value)
        self.plan_settle_sec = float(self.get_parameter("plan_settle_sec").value)
        self.retry_count = int(self.get_parameter("retry_count").value)
        self.retry_pause_sec = float(self.get_parameter("retry_pause_sec").value)
        self.max_box_count = int(self.get_parameter("max_box_count").value)

        self.arm_group = str(self.get_parameter("arm_group").value)
        self.gripper_group = str(self.get_parameter("gripper_group").value)

        self.cmd_distance_topic = str(self.get_parameter("cmd_distance_topic").value)
        self.backward_distance_m = float(self.get_parameter("backward_distance_m").value)
        self.distance_motion_extra_wait_sec = float(
            self.get_parameter("distance_motion_extra_wait_sec").value
        )

        self.wheel_radius = float(self.get_parameter("wheel_radius").value)
        self.steps_per_rev = int(self.get_parameter("steps_per_rev").value)
        self.microsteps = int(self.get_parameter("microsteps").value)
        self.distance_mode_sps = float(self.get_parameter("distance_mode_sps").value)

        self.steps_per_mech_rev = self.steps_per_rev * self.microsteps
        self.wheel_circumference = 2.0 * math.pi * self.wheel_radius
        self.meters_per_step = self.wheel_circumference / float(self.steps_per_mech_rev)

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

        # grab = [5, 56, 40, -79]
        self.declare_parameter("grab.base_rotating_waste_joint", 0.0873)
        self.declare_parameter("grab.rotating_waste_arm1_joint", 0.9774)
        self.declare_parameter("grab.arm1_arm2_joint", 0.6981)
        self.declare_parameter("grab.arm2_gripper_base_joint", -1.3788)

        # lift = [5, 45, 40, -81]
        self.declare_parameter("lift.base_rotating_waste_joint", 0.0873)
        self.declare_parameter("lift.rotating_waste_arm1_joint", 0.7854)
        self.declare_parameter("lift.arm1_arm2_joint", 0.6981)
        self.declare_parameter("lift.arm2_gripper_base_joint", -1.4137)

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

        # gripper
        self.declare_parameter("gripper_open.gripper_base_left_joint", -0.1)
        self.declare_parameter("gripper_close.gripper_base_left_joint", -0.9144)

        self.declare_parameter("restore_box_count", 3)

        self.restore_box_count = int(self.get_parameter("restore_box_count").value)
        self.placed_box_count = 0

        self.cmd_distance_pub = self.create_publisher(
            Float32MultiArray,
            self.cmd_distance_topic,
            10
        )

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
        self.get_logger().info(f"cmd_distance_topic: {self.cmd_distance_topic}")
        self.get_logger().info(
            f"distance estimate: meters_per_step={self.meters_per_step:.8f}, "
            f"distance_mode_sps={self.distance_mode_sps:.1f}"
        )

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

    def _sleep_with_cancel_check(self, goal_handle, duration_sec: float) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < duration_sec:
            if goal_handle is not None and goal_handle.is_cancel_requested:
                return False
            time.sleep(0.02)
        return True

    def estimate_distance_move_time(self, distance_m: float) -> float:
        step_count = abs(distance_m / self.meters_per_step)
        if self.distance_mode_sps <= 0.0:
            return 0.0
        return step_count / self.distance_mode_sps

    def move_base_by_distance(self, goal_handle, left_distance_m: float, right_distance_m: float) -> bool:
        msg = Float32MultiArray()
        msg.data = [float(left_distance_m), float(right_distance_m)]

        est_left = self.estimate_distance_move_time(left_distance_m)
        est_right = self.estimate_distance_move_time(right_distance_m)
        wait_time = max(est_left, est_right) + self.distance_motion_extra_wait_sec

        self.get_logger().info(
            f"Publishing cmd_distance: left={left_distance_m:.4f} m, "
            f"right={right_distance_m:.4f} m, estimated_wait={wait_time:.3f} s"
        )

        self.cmd_distance_pub.publish(msg)

        if not self._sleep_with_cancel_check(goal_handle, wait_time):
            return False

        return True

    def move_base_backward(self, goal_handle) -> bool:
        d = abs(self.backward_distance_m)
        return self.move_base_by_distance(goal_handle, -d, -d)

    def move_arm_joint_values(self, joint_values: dict, label: str) -> bool:
        try:
            time.sleep(self.plan_settle_sec)

            robot_state = RobotState(self.robot.get_robot_model())
            robot_state.joint_positions = joint_values

            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(robot_state=robot_state)

            plan_result = self.arm.plan()
            if not plan_result:
                self.get_logger().error(f"Planning failed for arm joint target '{label}'")
                return False

            self.robot.execute(plan_result.trajectory, controllers=["robot_arm_controller"])
            time.sleep(self.step_pause_sec)

            joint_str = ", ".join([f"{k}={v:.4f}" for k, v in joint_values.items()])
            self.get_logger().info(f"Executed arm target '{label}': {joint_str}")
            return True

        except Exception as exc:
            self.get_logger().error(f"Exception in move_arm_joint_values('{label}'): {exc}")
            return False

    def move_gripper_joint_values(self, joint_values: dict, label: str) -> bool:
        try:
            time.sleep(self.plan_settle_sec)

            robot_state = RobotState(self.robot.get_robot_model())
            robot_state.joint_positions = joint_values

            self.gripper.set_start_state_to_current_state()
            self.gripper.set_goal_state(robot_state=robot_state)

            plan_result = self.gripper.plan()
            if not plan_result:
                self.get_logger().error(f"Planning failed for gripper joint target '{label}'")
                return False

            self.robot.execute(plan_result.trajectory, controllers=["gripper_controller"])
            time.sleep(self.step_pause_sec)

            joint_str = ", ".join([f"{k}={v:.4f}" for k, v in joint_values.items()])
            self.get_logger().info(f"Executed gripper target '{label}': {joint_str}")
            return True

        except Exception as exc:
            self.get_logger().error(f"Exception in move_gripper_joint_values('{label}'): {exc}")
            return False

    def move_arm_joint_values_with_retry(self, joint_values: dict, label: str, retries: int = None) -> bool:
        if retries is None:
            retries = self.retry_count

        for attempt in range(1, retries + 1):
            self.get_logger().info(f"Arm target '{label}' attempt {attempt}/{retries}")
            if self.move_arm_joint_values(joint_values, label):
                return True

            if attempt < retries:
                self.get_logger().warn(
                    f"Retrying arm target '{label}' after failed attempt {attempt}/{retries}"
                )
                time.sleep(self.retry_pause_sec)

        self.get_logger().error(f"All retries failed for arm target '{label}'")
        return False

    def move_gripper_joint_values_with_retry(self, joint_values: dict, label: str, retries: int = None) -> bool:
        if retries is None:
            retries = self.retry_count

        for attempt in range(1, retries + 1):
            self.get_logger().info(f"Gripper target '{label}' attempt {attempt}/{retries}")
            if self.move_gripper_joint_values(joint_values, label):
                return True

            if attempt < retries:
                self.get_logger().warn(
                    f"Retrying gripper target '{label}' after failed attempt {attempt}/{retries}"
                )
                time.sleep(self.retry_pause_sec)

        self.get_logger().error(f"All retries failed for gripper target '{label}'")
        return False

    def open_gripper(self) -> bool:
        return self.move_gripper_joint_values_with_retry(
            self.get_gripper_joint_dict("gripper_open"),
            "gripper_open"
        )

    def close_gripper(self) -> bool:
        return self.move_gripper_joint_values_with_retry(
            self.get_gripper_joint_dict("gripper_close"),
            "gripper_close"
        )

    def fail_result(self, goal_handle, message: str):
        self.get_logger().error(message)
        result = PickBox.Result()
        goal_handle.abort()
        result.success = False
        result.message = message
        return result

    def success_result(self, goal_handle, message: str):
        self.get_logger().info(message)
        result = PickBox.Result()
        goal_handle.succeed()
        result.success = True
        result.message = message
        return result

    def canceled_result(self, goal_handle, message: str):
        self.get_logger().warn(message)
        result = PickBox.Result()
        goal_handle.canceled()
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

        if not self.move_arm_joint_values_with_retry(home_joints, "home"):
            self.get_logger().warn("Startup home failed")

        if not self.open_gripper():
            self.get_logger().warn("Startup gripper_open failed")

    def get_current_place_prefix(self) -> str:
        pose_index = min(self.placed_box_count + 1, self.max_box_count)
        return f"place{pose_index}"

    def do_pick_place_sequence(self, goal_handle, side: str, place_prefix: str):
        home_joints = self.get_arm_joint_dict("home")
        grab_joints = self.get_arm_joint_dict("grab")
        lift_joints = self.get_arm_joint_dict("lift")
        place_joints = self.get_arm_joint_dict(place_prefix)

        self.publish_feedback(goal_handle, "gripper_open", 0.10)
        if not self.open_gripper():
            return self.fail_result(goal_handle, "Failed at step: gripper_open")
        if goal_handle.is_cancel_requested:
            return self.canceled_result(goal_handle, "Goal canceled after gripper_open")

        self.publish_feedback(goal_handle, "move_grab_pose", 0.25)
        if not self.move_arm_joint_values_with_retry(grab_joints, "grab"):
            return self.fail_result(goal_handle, "Failed at step: grab")
        if goal_handle.is_cancel_requested:
            return self.canceled_result(goal_handle, "Goal canceled after grab")

        self.publish_feedback(goal_handle, "gripper_close", 0.40)
        if not self.close_gripper():
            return self.fail_result(goal_handle, "Failed at step: gripper_close")
        if goal_handle.is_cancel_requested:
            return self.canceled_result(goal_handle, "Goal canceled after gripper_close")

        self.publish_feedback(goal_handle, "move_lift_pose", 0.55)
        if not self.move_arm_joint_values_with_retry(lift_joints, "lift"):
            return self.fail_result(goal_handle, "Failed at step: lift")
        if goal_handle.is_cancel_requested:
            return self.canceled_result(goal_handle, "Goal canceled after lift")

        self.publish_feedback(goal_handle, "move_base_backward", 0.70)
        if not self.move_base_backward(goal_handle):
            return self.fail_result(goal_handle, "Failed at step: move_base_backward")
        if goal_handle.is_cancel_requested:
            return self.canceled_result(goal_handle, "Goal canceled after move_base_backward")

        self.publish_feedback(goal_handle, "move_place_pose", 0.85)
        if not self.move_arm_joint_values_with_retry(place_joints, place_prefix):
            return self.fail_result(goal_handle, f"Failed at step: {place_prefix}")
        if goal_handle.is_cancel_requested:
            return self.canceled_result(goal_handle, "Goal canceled after place pose")

        self.publish_feedback(goal_handle, "release_box", 0.95)
        if not self.open_gripper():
            return self.fail_result(goal_handle, "Failed at step: release gripper_open")
        if goal_handle.is_cancel_requested:
            return self.canceled_result(goal_handle, "Goal canceled after release")

        self.publish_feedback(goal_handle, "return_home", 1.00)
        if not self.move_arm_joint_values_with_retry(home_joints, "home"):
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
            if not self.move_arm_joint_values_with_retry(place_joints, place_prefix):
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
            if not self.move_arm_joint_values_with_retry(restore_joints, "restore"):
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
        if not self.move_arm_joint_values_with_retry(home_joints, "home"):
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