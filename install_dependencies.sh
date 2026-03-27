#!/usr/bin/env bash
# =============================================================================
# SLRC-2026 Dependency Installer
# Scanned from all packages in src/
# =============================================================================
set -e

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Detect ROS 2 distro ───────────────────────────────────────────────────────
if [ -n "$ROS_DISTRO" ]; then
    ROS_DISTRO_NAME="$ROS_DISTRO"
else
    # Fallback: pick the first installed distro under /opt/ros
    ROS_DISTRO_NAME=$(ls /opt/ros 2>/dev/null | head -1)
fi

if [ -z "$ROS_DISTRO_NAME" ]; then
    error "ROS 2 does not appear to be installed. Install it first:\n  https://docs.ros.org/en/rolling/Installation.html"
fi
info "Detected ROS 2 distro: $ROS_DISTRO_NAME"

# ── Helpers ───────────────────────────────────────────────────────────────────
apt_install() {
    info "APT: installing $*"
    sudo apt-get install -y "$@"
}

pip_install() {
    info "pip: installing $*"
    pip3 install --break-system-packages "$@" 2>/dev/null \
        || pip3 install "$@"
}

# ── 1. System dependencies (apt) ─────────────────────────────────────────────
info "=== Step 1/4: Updating apt package lists ==="
sudo apt-get update -y

info "=== Step 2/4: Installing system apt packages ==="
apt_install \
    python3-pip \
    python3-opencv \
    python3-numpy \
    python3-smbus \
    python3-smbus2 \
    python3-libgpiod \
    python3-yaml \
    python3-pytest \
    i2c-tools

# ── 2. ROS 2 packages (apt) ───────────────────────────────────────────────────
ROS="ros-${ROS_DISTRO_NAME}"

info "=== Step 3/4: Installing ROS 2 packages ==="

# Core ROS 2 Python client
apt_install \
    "${ROS}-rclpy" \
    "${ROS}-std-msgs" \
    "${ROS}-geometry-msgs" \
    "${ROS}-sensor-msgs" \
    "${ROS}-action-msgs"

# Vision / camera bridge
apt_install \
    "${ROS}-cv-bridge" \
    "${ROS}-image-transport"

# Robot description / state publishing
apt_install \
    "${ROS}-robot-state-publisher" \
    "${ROS}-joint-state-publisher" \
    "${ROS}-joint-state-publisher-gui" \
    "${ROS}-xacro" \
    "${ROS}-tf2-ros" \
    "${ROS}-tf2-tools"

# ROS 2 launch infrastructure
apt_install \
    "${ROS}-launch" \
    "${ROS}-launch-ros" \
    "${ROS}-ament-index-python"

# ROS 2 interface generation (robot_arm_interfaces)
apt_install \
    "${ROS}-rosidl-default-generators" \
    "${ROS}-rosidl-default-runtime" \
    "${ROS}-action-tutorials-interfaces"

# RViz2
apt_install \
    "${ROS}-rviz2" \
    "${ROS}-rviz-common" \
    "${ROS}-rviz-default-plugins"

# MoveIt 2
apt_install \
    "${ROS}-moveit" \
    "${ROS}-moveit-py" \
    "${ROS}-moveit-configs-utils" \
    "${ROS}-moveit-ros-move-group" \
    "${ROS}-moveit-ros-visualization" \
    "${ROS}-moveit-ros-warehouse" \
    "${ROS}-moveit-kinematics" \
    "${ROS}-moveit-planners" \
    "${ROS}-moveit-simple-controller-manager" \
    "${ROS}-moveit-setup-assistant"

# ros2_control
apt_install \
    "${ROS}-controller-manager"

# Warehouse backend for MoveIt
apt_install \
    "${ROS}-warehouse-ros-mongo" 2>/dev/null \
    || warn "warehouse-ros-mongo not available for $ROS_DISTRO_NAME – skipping"

# Webots ROS 2 driver  (simulation)
apt_install \
    "${ROS}-webots-ros2-driver" 2>/dev/null \
    || warn "webots-ros2-driver not available for $ROS_DISTRO_NAME – you may need to install Webots manually."

# ── 3. Python (pip) packages ──────────────────────────────────────────────────
info "=== Step 4/4: Installing Python pip packages ==="

# Computer vision
pip_install \
    opencv-python \
    numpy \
    matplotlib

# Adafruit CircuitPython stack (servo, ToF, OLED, GPIO)
pip_install \
    adafruit-blinka \
    adafruit-circuitpython-servokit \
    adafruit-circuitpython-vl53l0x \
    adafruit-circuitpython-ssd1306 \
    Pillow

# I2C / SMBus for MPU-6050
pip_install \
    smbus2

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
info "============================================================"
info " All dependencies installed successfully!"
info " Source your ROS workspace before building:"
info "   source /opt/ros/${ROS_DISTRO_NAME}/setup.bash"
info "   cd /home/thunderbot/SLRC-2026-b && colcon build"
info "============================================================"
