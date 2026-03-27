from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="arm_pose_tuner",
            executable="pose_tuner",
            name="pose_tuner",
            output="screen",
            parameters=[
                {"arm_group": "robot_arm"},
                {"gripper_group": "gripper"},
                {"robot_name": "my_arm"},
                {"moveit_config_package": "robot_arm_v2_moveit_config"},
            ],
        )
    ])