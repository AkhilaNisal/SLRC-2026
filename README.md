# 🤖 SLRC-2026 Autonomous Mobile Manipulation Robot

<p align="center">
  <img src="images/robot_banner.png" width="800"/>
</p>

## 🚀 Overview

This project is developed for the **Sri Lanka Robotics Challenge (SLRC) 2026**, organized by the **University of Moratuwa, Sri Lanka**.

It is a fully autonomous **mobile manipulation robot system** built using **ROS 2**, capable of performing navigation, object detection, and robotic arm manipulation tasks in a structured competition environment.

The system integrates:
- Autonomous navigation
- Computer vision-based object detection
- Camera-based path tracking
- 5-DOF robotic arm manipulation
- MoveIt 2 motion planning
- Real-time sensor fusion

---

## 🎥 Demonstrations

### 🦾 Real Robot Demo
<p align="center">
  <img src="images/real_robot.jpg" width="700"/>
</p>

https://github.com/user-attachments/assets/real_robot_demo.mp4

---

### 🧪 Simulation Demo (Webots)
<p align="center">
  <img src="images/simulation.png" width="700"/>
</p>

https://github.com/user-attachments/assets/simulation_demo.mp4

---

### 🦾 MoveIt Arm Control
<p align="center">
  <img src="images/moveit_rviz.png" width="700"/>
</p>

https://github.com/user-attachments/assets/arm_demo.mp4

---

## ✨ Key Features

### 🤖 Autonomous Mobile Robot
- Differential drive platform
- NEMA17 stepper motor control
- Path tracking and obstacle avoidance
- Fully autonomous mission execution

### 🦾 5-DOF Robotic Arm
- Custom-designed robotic manipulator
- MG996R + SG90 servo combination
- MoveIt 2 motion planning
- Pick-and-place operations
- Collision-aware trajectory execution

### 👁️ Computer Vision System
- USB camera based perception
- Color-based object detection
- Object tracking
- Path tracking
- AprilTag detection
- Visual servoing

### 🧠 Intelligent Task System
- Autonomous task execution (Task 1 / Task 2)
- Object search and alignment
- Navigation + manipulation coordination
- Action server-based architecture

---

## 🔩 Hardware Overview

- Raspberry Pi (Main Controller)
- USB Camera (Vision System)
- 2 × NEMA17 Stepper Motors (Drive System)
- MG996R + SG90 Servo Motors (Robot Arm)
- 16-bit Servo Driver Shield
- ToF Distance Sensors
- MPU6050 IMU Sensor
- OLED Display Module

---

## 💻 Software Stack

- ROS 2 (Robot middleware)
- MoveIt 2 (Motion planning)
- Webots (Simulation)
- OpenCV (Computer Vision)
- SolidWorks (Mechanical design)
- Python & C++ (Implementation)

---

## 📁 Repository Structure

src/
├── robocop_pkg
├── camera_feed
├── color_tracker_control
├── apriltag_decoder
├── stepper_control
├── tof_sensors
├── mpu6050_ros2
├── oled_display
├── robot_arm
├── robot_arm_v2
├── robot_arm_moveit_config
├── robot_arm_v2_moveit_config
├── robot_arm_interfaces
├── robot_arm_bringup
├── rpi_arm_hardware
├── arm_pose_tuner
└── robot_test

## 🧩 Package Overview

robocop_pkg:
Core intelligence for task execution, mission control, navigation coordination, and action server communication.

camera_feed:
USB camera streaming and ROS image publishing.

color_tracker_control:
Color segmentation, object tracking, and vision-based path alignment.

apriltag_decoder:
AprilTag detection and pose estimation.

robot_arm:
URDF and kinematic model of 5-DOF arm.

robot_arm_v2:
Improved arm geometry and kinematics.

robot_arm_moveit_config:
MoveIt 2 motion planning configuration.

robot_arm_v2_moveit_config:
Upgraded MoveIt configuration.

robot_arm_interfaces:
Custom ROS 2 messages, services, and actions.

robot_arm_bringup:
Launch system for robot arm and MoveIt.

rpi_arm_hardware:
Hardware interface for Raspberry Pi servo control.

stepper_control:
Controls NEMA17 motors for robot movement.

tof_sensors:
Distance sensing and obstacle detection.

mpu6050_ros2:
IMU integration (acceleration, gyro, orientation).

oled_display:
Robot status display system.

arm_pose_tuner:
Calibration and tuning tools.

robot_test:
Testing and debugging utilities.

## 🧭 System Architecture

Camera + Sensors
      ↓
Computer Vision Layer
      ↓
Task Manager (robocop_pkg)
      ↓
Navigation System → Mobile Base
Manipulation System → 5-DOF Arm
      ↓
Autonomous SLRC Tasks

## ⚙️ Build Instructions

mkdir -p ~/slrc_ws/src
cd ~/slrc_ws
git clone https://github.com/AkhilaNisal/SLRC-2026.git src/SLRC-2026
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash

## ▶️ Run Instructions

colcon build --symlink-install
source install/setup.bash

ros2 launch robot_arm_bringup bringup.launch.py
ros2 launch camera_feed camera.launch.py
ros2 launch robot_arm_moveit_config moveit.launch.py

ros2 run robocop_pkg task1
ros2 run robocop_pkg task2

## 🏁 Competition

Sri Lanka Robotics Challenge (SLRC) 2026  
University of Moratuwa, Sri Lanka

## 👨‍💻 Authors

SLRC 2026 Robotics Team

## 📜 License

Educational and research purposes only.
