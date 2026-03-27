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
        output='screen'
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

    robot_arm_centering_action_server = Node(
        package='robocop_pkg',
        executable='robot_arm_centering_action_server',
        name='robot_arm_centering_action_server',
        output='screen',
    )

    task1 = Node(
        package='robocop_pkg',
        executable='task1',
        name='task1',
        output='screen',
    )

    # Delay task nodes to allow hardware nodes to initialize first
    delayed_task_nodes = TimerAction(
        period=8.0,
        actions=[
            robot_arm_centering_action_server,
            task1,
        ]
    )

    return LaunchDescription([
        mpu_node,
        camera_feed_node,
        tof_node,
        cmd_vel_stepper_node,
        delayed_task_nodes,
    ])
