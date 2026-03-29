from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():

    mpu_node = Node(
        package='mpu6050_ros2',
        executable='mpu6050_node',
        name='mpu6050_node',
        output='screen',
        parameters=[{
            'i2c_bus': 1,
            'i2c_address': 0x68,
            'frame_id': 'imu_link',
            'publish_rate': 50.0,
            'stationary_gyro_threshold_dps': 0.8,
            'stationary_accel_threshold_g': 0.08,
            'yaw_bias_adapt_alpha': 0.001,
        }]
    )

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

    task1_hardcoded = Node(
        package='robocop_pkg',
        executable='task1_hardcoded',
        name='task1_hardcoded',
        output='screen',
        parameters=[{
            'heading_weight_both': 0.15,
            'heading_weight_missing': 0.80,
            'imu_fusion_alpha': 0.05,
        }],
    )

    # Delay task node to allow hardware nodes to initialize first
    delayed_task_nodes = TimerAction(
        period=6.0,
        actions=[task1_hardcoded],
    )

    return LaunchDescription([
        mpu_node,
        camera_feed_node,
        apriltag_decoder_node,
        tof_node,
        cmd_vel_stepper_node,
        delayed_task_nodes,
    ])
