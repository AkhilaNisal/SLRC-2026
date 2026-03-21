#!/usr/bin/env python3
import os
import threading
import time
import yaml

import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory

from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanningComponent
from moveit_configs_utils import MoveItConfigsBuilder


class PoseTuner(Node):
    def __init__(self):
        super().__init__("pose_tuner")

        self.declare_parameter("arm_group", "robot_arm")
        self.declare_parameter("gripper_group", "gripper")
        self.declare_parameter("moveit_config_package", "robot_arm_v2_moveit_config")
        self.declare_parameter("robot_name", "my_arm")
        self.declare_parameter("poses_file", "")
        self.declare_parameter("step_pause_sec", 0.5)
        self.declare_parameter("enable_gripper", True)

        self.arm_group = str(self.get_parameter("arm_group").value)
        self.gripper_group = str(self.get_parameter("gripper_group").value)
        self.moveit_config_package = str(self.get_parameter("moveit_config_package").value)
        self.robot_name = str(self.get_parameter("robot_name").value)
        self.step_pause_sec = float(self.get_parameter("step_pause_sec").value)
        self.enable_gripper = bool(self.get_parameter("enable_gripper").value)
        poses_file_param = str(self.get_parameter("poses_file").value)

        if poses_file_param:
            self.poses_file = poses_file_param
        else:
            pkg_share = get_package_share_directory("arm_pose_tuner")
            self.poses_file = os.path.join(pkg_share, "config", "arm_poses.yaml")

        self.arm_joint_names = [
            "base_rotating_waste_joint",
            "rotating_waste_arm1_joint",
            "arm1_arm2_joint",
            "arm2_gripper_base_joint",
        ]

        # Build MoveIt config properly, including planning pipelines.
        moveit_config = (
            MoveItConfigsBuilder(
                robot_name=self.robot_name,
                package_name=self.moveit_config_package,
            )
            .robot_description_semantic("config/my_arm.srdf", {"name": self.robot_name})
            .robot_description_kinematics()
            .planning_pipelines(
                pipelines=["ompl"],
                default_planning_pipeline="ompl",
            )
            .to_dict()
        )

        # Add some explicit config in case the package is minimal.
        moveit_config["planning_scene_monitor"] = {
            "name": "planning_scene_monitor",
            "robot_description": "robot_description",
            "joint_state_topic": "/joint_states",
            "attached_collision_object_topic": "/moveit_cpp/planning_scene_monitor",
            "publish_planning_scene_topic": "/moveit_cpp/publish_planning_scene",
            "monitored_planning_scene_topic": "/moveit_cpp/monitored_planning_scene",
            "wait_for_initial_state_timeout": 10.0,
        }

        moveit_config["planning_pipelines"] = {
            "pipeline_names": ["ompl"],
            "default_planning_pipeline": "ompl",
        }

        moveit_config["plan_request_params"] = {
            "planning_attempts": 1,
            "planning_pipeline": "ompl",
            "max_velocity_scaling_factor": 1.0,
            "max_acceleration_scaling_factor": 1.0,
        }

        moveit_config["ompl"] = {
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
            "start_state_max_bounds_error": 0.1,
        }

        self.get_logger().info("Initializing MoveItPy...")
        self.robot = MoveItPy(node_name="moveit_py_pose_tuner", config_dict=moveit_config)
        self.arm: PlanningComponent = self.robot.get_planning_component(self.arm_group)

        self.gripper = None
        if self.enable_gripper:
            try:
                self.gripper = self.robot.get_planning_component(self.gripper_group)
                self.get_logger().info(f"Gripper planning component ready: {self.gripper_group}")
            except Exception as exc:
                self.gripper = None
                self.get_logger().warn(f"Gripper component unavailable: {exc}")

        self.poses = self.load_poses()

        self._stop_requested = False
        self._cmd_thread = threading.Thread(target=self.command_loop, daemon=True)
        self._cmd_thread.start()

        self.get_logger().info("Pose tuner ready.")
        self.print_help()

    def print_help(self):
        print("\nCommands:")
        print("  help")
        print("  list")
        print("  show")
        print("  move <pose_name>")
        print("  set <pose_name>")
        print("  save")
        print("  jog <joint_name> <delta>")
        print("  open")
        print("  close")
        print("  quit\n")

    def load_poses(self):
        if not os.path.exists(self.poses_file):
            self.get_logger().warn(f"No poses file found at {self.poses_file}. Starting empty.")
            return {"poses": {}}

        with open(self.poses_file, "r") as f:
            data = yaml.safe_load(f) or {}

        if "poses" not in data:
            data["poses"] = {}
        return data

    def save_poses(self):
        os.makedirs(os.path.dirname(self.poses_file), exist_ok=True)
        with open(self.poses_file, "w") as f:
            yaml.safe_dump(self.poses, f, sort_keys=False)
        self.get_logger().info(f"Saved poses to {self.poses_file}")

    def list_poses(self):
        pose_names = sorted(self.poses.get("poses", {}).keys())
        if not pose_names:
            print("No saved poses.")
            return
        print("Saved poses:")
        for name in pose_names:
            print(f"  - {name}")

    def get_current_joint_values(self):
        try:
            state = self.arm.get_start_state()
        except Exception:
            state = None

        values = {}
        if state is not None:
            for joint in self.arm_joint_names:
                try:
                    positions = state.get_joint_group_positions(joint)
                    if positions:
                        values[joint] = float(positions[0])
                        continue
                except Exception:
                    pass

        # Fallback: ask planning scene current state
        try:
            scene_monitor = self.robot.get_planning_scene_monitor()
            scene = scene_monitor.get_planning_scene()
            current_state = scene.current_state
            for joint in self.arm_joint_names:
                if joint in values:
                    continue
                try:
                    positions = current_state.get_joint_positions(joint)
                    if positions:
                        values[joint] = float(positions[0])
                except Exception:
                    pass
        except Exception:
            pass

        for joint in self.arm_joint_names:
            values.setdefault(joint, 0.0)

        return values

    def move_arm_joint_values(self, joint_values: dict, label: str) -> bool:
        try:
            robot_state = RobotState(self.robot.get_robot_model())
            robot_state.joint_positions = joint_values

            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(robot_state=robot_state)

            plan_result = self.arm.plan()
            if not plan_result:
                self.get_logger().error(f"Planning failed for '{label}'")
                return False

            self.robot.execute(plan_result.trajectory, controllers=[])
            time.sleep(self.step_pause_sec)

            joint_str = ", ".join(f"{k}={v:.4f}" for k, v in joint_values.items())
            self.get_logger().info(f"Executed '{label}': {joint_str}")
            return True

        except Exception as exc:
            self.get_logger().error(f"Exception while moving to '{label}': {exc}")
            return False

    def move_gripper_named(self, name: str) -> bool:
        if self.gripper is None:
            self.get_logger().warn("Gripper planning component is unavailable.")
            return False

        try:
            self.gripper.set_start_state_to_current_state()
            self.gripper.set_goal_state(configuration_name=name)

            plan_result = self.gripper.plan()
            if not plan_result:
                self.get_logger().error(f"Planning failed for gripper target '{name}'")
                return False

            self.robot.execute(plan_result.trajectory, controllers=[])
            time.sleep(self.step_pause_sec)
            self.get_logger().info(f"Executed gripper target '{name}'")
            return True

        except Exception as exc:
            self.get_logger().error(f"Gripper move failed for '{name}': {exc}")
            return False

    def jog_joint(self, joint_name: str, delta: float):
        if joint_name not in self.arm_joint_names:
            self.get_logger().error(f"Unknown joint '{joint_name}'")
            return

        current = self.get_current_joint_values()
        current[joint_name] += delta
        self.move_arm_joint_values(current, f"jog_{joint_name}")

    def command_loop(self):
        while rclpy.ok() and not self._stop_requested:
            try:
                line = input("pose_tuner> ").strip()
            except EOFError:
                self._stop_requested = True
                break
            except KeyboardInterrupt:
                self._stop_requested = True
                break

            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()

            try:
                if cmd == "help":
                    self.print_help()

                elif cmd == "list":
                    self.list_poses()

                elif cmd == "show":
                    current = self.get_current_joint_values()
                    print(current)

                elif cmd == "move" and len(parts) == 2:
                    pose_name = parts[1]
                    pose = self.poses.get("poses", {}).get(pose_name)
                    if not pose:
                        self.get_logger().error(f"Pose '{pose_name}' not found")
                        continue
                    self.move_arm_joint_values(pose, pose_name)

                elif cmd == "set" and len(parts) == 2:
                    pose_name = parts[1]
                    current = self.get_current_joint_values()
                    self.poses.setdefault("poses", {})[pose_name] = current
                    self.get_logger().info(f"Captured current pose as '{pose_name}'")

                elif cmd == "save":
                    self.save_poses()

                elif cmd == "jog" and len(parts) == 3:
                    joint_name = parts[1]
                    delta = float(parts[2])
                    self.jog_joint(joint_name, delta)

                elif cmd == "open":
                    self.move_gripper_named("gripper_open")

                elif cmd == "close":
                    self.move_gripper_named("gripper_close")

                elif cmd in ("quit", "exit"):
                    self._stop_requested = True
                    rclpy.shutdown()
                    break

                else:
                    self.get_logger().info("Unknown command. Type 'help'.")

            except Exception as exc:
                self.get_logger().error(f"Command failed: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = PoseTuner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_requested = True
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()