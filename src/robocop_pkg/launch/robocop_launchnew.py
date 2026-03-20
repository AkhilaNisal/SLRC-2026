from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
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

    task_manager = Node(
        package='robocop_pkg',
        executable='task_manager',
        name='task_manager',
        output='screen',
        parameters=[{
            'task2_package': 'robocop_pkg',
            'task2_executable': 'task2_with_arm',
            'task3_package': 'robocop_pkg',
            'task3_executable': 'task3',
            'task2_status_topic': '/task2/status',
            'task3_status_topic': '/task3/status',
            'startup_delay_sec': 2.0,
            'shutdown_wait_sec': 3.0,
        }]
    )

    tof_dual_node = Node(
        package='tof_sensors',
        executable='tof_dual_node',
        name='tof_dual_node',
        output='screen',
        parameters=[{
            'left_range_topic': '/robocop/ds_left',
            'right_range_topic': '/robocop/ds_right',
            'left_frame_id': 'tof_left',
            'right_frame_id': 'tof_right',
            'publish_rate_hz': 10.0,
            'left_xshut_pin': 'D17',
            'right_xshut_pin': 'D27',
            'left_i2c_address': 0x30,
            'right_i2c_address': 0x29,
        }]
    )

    task2_with_arm = Node(
        package='robocop_pkg',
        executable='task2_with_arm',   # must match setup.py entry point
        name='task2_with_arm',
        output='screen',
    
    )

    task3 = Node(
        package='robocop_pkg',
        executable='task3',   # must match setup.py entry point
        name='task3',
        output='screen',
    
    )

    return LaunchDescription([
        camera_feed_node,
        tof_dual_node,
        robot_arm_action_server,
        cmd_vel_stepper_node,
        # task2_with_arm,
        task3,
        # task_masnager,
    ])