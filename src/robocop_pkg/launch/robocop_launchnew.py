from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    moveit_pkg_share = get_package_share_directory('robot_arm_v2_moveit_config')
    moveit_demo_launch = os.path.join(moveit_pkg_share, 'launch', 'demo.launch.py')

    pca9685_bridge = ExecuteProcess(
        cmd=[
            '/home/thunderbot/SLRC-2026/venv/bin/python',
            '/home/thunderbot/SLRC-2026/src/rpi_arm_hardware/scripts/pca9685_bridge.py'
        ],
        name='pca9685_bridge',
        output='screen'
    )

    moveit_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(moveit_demo_launch)
    )

    camera_feed_node = Node(
        package='camera_feed',
        executable='camera_feed_node',
        name='camera_feed_node',
        output='screen'
    )

    cmd_vel_stepper_node = Node(
        package='stepper_control',
        executable='cmd_vel_stepper_node',
        name='cmd_vel_stepper_node',
        output='screen',
        parameters=[{
            'wheel_radius': 0.0325,
            'wheel_base': 0.20,
            'steps_per_rev': 200,
            'microsteps': 16,
            'max_steps_per_sec': 4000.0,
            'accel_steps_per_sec2': 3500.0,
            'decel_steps_per_sec2': 3500.0,
            'cmd_vel_timeout': 0.2,
            'chip_name': 'gpiochip4',
            'left_en_pin': 22,
            'left_dir_pin': 23,
            'left_step_pin': 24,
            'right_en_pin': 12,
            'right_dir_pin': 5,
            'right_step_pin': 6,
            'enable_active_low': True,
            'left_dir_inverted': False,
            'right_dir_inverted': True,
            'cmd_vel_topic': '/cmd_vel',
        }]
    )

    robot_arm_action_server = Node(
        package='robocop_pkg',
        executable='robot_arm_action_server',
        name='robot_arm_action_server',
        output='screen',
        parameters=[{
            'action_name': '/pick_box',
            'startup_delay_sec': 1.0,
            'step_pause_sec': 0.5,
            'max_box_count': 6,
            'arm_group': 'robot_arm',
            'gripper_group': 'gripper',
            'restore_box_count': 3,
        }]
    )

    tof_node = Node(
        package='tof_sensors',
        executable='tof_node',
        name='tof_node',
        output='screen',
    )

    task2_with_arm = Node(
        package='robocop_pkg',
        executable='task2_with_arm',   # must match setup.py entry point
        name='task2_with_arm',
        output='screen',
    
    )

    task3 = Node(
        package='robocop_pkg',
        executable='task3',
        name='task3',
        output='screen',
    )

    delayed_task_nodes = TimerAction(
        period=8.0,
        actions=[
            robot_arm_action_server,
            # task3,
            task2_with_arm,
        ]
    )

    return LaunchDescription([
        pca9685_bridge,
        moveit_demo,
        camera_feed_node,
        tof_node,
        cmd_vel_stepper_node,
        delayed_task_nodes,
    ])