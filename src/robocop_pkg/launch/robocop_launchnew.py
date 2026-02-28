import os
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController


def generate_launch_description():
    package_dir = get_package_share_directory('robocop_pkg')
    urdf_path = os.path.join(package_dir, 'resource', 'robocop.urdf')
    world_path = os.path.join(package_dir, 'worlds', 'arena1.wbt')

    # Read URDF file content (robot_description should be the XML string)
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    # Add a launch argument for mode, so you can run: mode:=realtime / fast / headless / pause
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='realtime',
        description='Webots simulation mode: realtime, fast, headless, pause'
    )
    mode = LaunchConfiguration('mode')

    webots = WebotsLauncher(
        world=world_path,
        mode=mode,
        ros2_supervisor=True
    )

    my_robot_driver = WebotsController(
        robot_name='robocop',
        parameters=[
            {'robot_description': robot_description},
        ],
        # Every time one resets the simulation the controller is automatically respawned
        respawn=True
    )

    obstacle_avoider = Node(
        package='robocop_pkg',
        executable='white_line_follower',
        output='screen'
    )

    return LaunchDescription([
        mode_arg,

        webots,
        webots._supervisor,   # ✅ REQUIRED when ros2_supervisor=True

        my_robot_driver,
        # obstacle_avoider,

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=webots,
                on_exit=[EmitEvent(event=Shutdown())],
            )
        )
    ])