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

    robot_arm_centering_action_server = Node(
        package='robocop_pkg',
        executable='robot_arm_centering_action_server',
        name='robot_arm_centering_action_server',
        output='screen',

    )

    tof_node = Node(
        package='tof_sensors',
        executable='tof_node',
        name='tof_node',
        output='screen',
    ) 

    task1 = Node(
        package='robocop_pkg',
        executable='task1',
        name='task1',
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

    mpu_node = Node(
        package='mpu6050_ros2',
        executable='mpu6050_node',
        name='mpu6050_node',
        output='screen',
        parameters=[
            {
                'i2c_bus': 1,
                'i2c_address': 0x68,
                'frame_id': 'imu_link',
                'publish_rate': 50.0,
                'stationary_gyro_threshold_dps': 0.8,
                'stationary_accel_threshold_g': 0.08,
                'yaw_bias_adapt_alpha': 0.001,
            }
        ]
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

    delayed_task_nodes = TimerAction(
        period=8.0,
        actions=[
            # robot_arm_action_server,
            robot_arm_centering_action_server,
            # task3,
            # task2_with_arm,
            # task_manager,
            task1,


        ]
    )

   

    return LaunchDescription([
        mpu_node,
        # pca9685_bridge,
        # moveit_demo,
        camera_feed_node,
        tof_node,
        cmd_vel_stepper_node,
        delayed_task_nodes,
    ])