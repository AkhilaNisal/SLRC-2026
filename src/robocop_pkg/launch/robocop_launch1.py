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
    world_path = os.path.join(package_dir, 'worlds', 'arena2.wbt')

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
        
    white_line_follower = Node(
        package='robocop_pkg',                 # ✅ REQUIRED
        executable='white_line_follower',      # must exist in setup.py entry_points
        name='white_line_follower',
        output='screen',
        parameters=[{
            'image_topic': '/camera/image/image_color',
            'cmd_vel_topic': '/cmd_vel',
            'linear_speed': 0.15,
            'kp': 0.004,
            'roi_y_start': 0.60,
            'min_area': 5000,
            'h_low': 0, 's_low': 0, 'v_low': 180,
            'h_high': 180, 's_high': 70, 'v_high': 255,
        }]
    )

    perspective_rectifier = Node(
        package='robocop_pkg',
        executable='perspective_rectifier',
        name='perspective_rectifier',
        output='screen',
        parameters=[{
            'input_image_topic': '/camera/image/image_color',
            'output_image_topic': '/camera/image_rect',
            'homography_topic': '/camera/homography',
            'output_width': 640,
            'output_height': 480,

            # top-left
            'src_tl_x': 300.0,
            'src_tl_y': 300.0,

            # top-right
            'src_tr_x': 340.0,
            'src_tr_y': 300.0,

            # bottom-right
            'src_br_x': 620.0,
            'src_br_y': 460.0,

            # bottom-left
            'src_bl_x': 40.0,
            'src_bl_y': 460.0,
        }]
    )



    red_box_perpendicular_seeker = Node(
        package='robocop_pkg',
        executable='red_box_perpendicular_seeker',   # must match setup.py entry point
        name='red_box_perpendicular_seeker',
        output='screen'
    )


    task2 = Node(
        package='robocop_pkg',
        executable='task2',   # must match setup.py entry point
        name='task2',
        output='screen',
    
    )
    task2_new = Node(
        package='robocop_pkg',
        executable='task2_new',   # must match setup.py entry point
        name='task2_new',
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
        executable='task3',   # must match setup.py entry point
        name='task3',
        output='screen',
    
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



    return LaunchDescription([
        mode_arg,

        webots,
        webots._supervisor,   # ✅ REQUIRED when ros2_supervisor=True

        my_robot_driver,
        # white_line_follower,
        # perspective_rectifier,
        # red_box_perpendicular_seeker,
        # task2_new,
        # task2_with_arm,
        task3,
        # task_manager,


        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=webots,
                on_exit=[EmitEvent(event=Shutdown())],
            )
        )
    ])