from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():

    camera_feed_node = Node(
        package='camera_feed',
        executable='camera_feed_node',
        name='camera_feed_node',
        output='screen',
    )

    apriltag_decoder_node = Node(
        package='apriltag_decoder',
        executable='apriltag_decoder_node',
        name='apriltag_decoder_node',
        output='screen',
        parameters=[{
            'required_unique_tags': 8,
            'families': 'tagStandard52h13',
            'publish_on_each_detection': True,
            'publish_debug_image': True,
        }],
    )

    tof_node = Node(
        package='tof_sensors',
        executable='tof_node',
        name='tof_node',
        output='screen',
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

    task1 = Node(
        package='robocop_pkg',
        executable='task1',
        name='task1',
        output='screen',
        parameters=[{
            # Steppers track straight — disable gyro heading-hold
            'heading_weight_both': 0.0,
            'heading_weight_missing': 0.0,
        }],
    )

    # Delay task1 to allow hardware nodes to initialize first
    delayed_task_nodes = TimerAction(
        period=4.0,
        actions=[task1],
    )

    return LaunchDescription([
        camera_feed_node,
        apriltag_decoder_node,
        tof_node,
        cmd_vel_stepper_node,
        delayed_task_nodes,
    ])
