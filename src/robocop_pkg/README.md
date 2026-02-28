# robocop_pkg

ROS 2 package for the RoboCop robot system, designed for simulation and autonomous control using Webots.

This package is implemented in **Python (rclpy)** and integrates with Webots for robot simulation.

---

## 📦 Package Overview

`robocop_pkg` provides:

- Robot control logic
- Webots driver integration
- Sensor data processing
- Computer vision support via OpenCV
- Geometry-based motion commands

---

## 🧰 Dependencies

This package depends on the following ROS 2 packages:

| Dependency | Purpose |
|------------|----------|
| `rclpy` | ROS 2 Python client library |
| `geometry_msgs` | Velocity and pose message types |
| `sensor_msgs` | Sensor message types (e.g., images) |
| `cv_bridge` | Convert ROS images to OpenCV format |
| `webots_ros2_driver` | Interface between ROS 2 and Webots |

---

## 🖥 System Requirements

- Ubuntu 24.04
- ROS 2 Jazzy
- Webots (compatible version)
- Python 3.10+

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/AkhilaNisal/SLRC-2026.git
cd SLRC-2026
