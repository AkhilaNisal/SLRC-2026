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

# ── Step 0: Install ROS 2 Jazzy ───────────────────────────────────────────────
info "=== Step 0: Installing ROS 2 Jazzy ==="

# Locale
sudo apt-get install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Universe repo
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y universe

# ROS 2 apt repository
sudo apt-get update -y
sudo apt-get install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt-get update -y
sudo apt-get upgrade -y

# Install ROS 2 Jazzy Desktop (full)
sudo apt-get install -y ros-jazzy-desktop

# Dev tools
sudo apt-get install -y \
    python3-rosdep \
    python3-colcon-common-extensions \
    python3-vcstool \
    ros-dev-tools

# Initialise rosdep (skip if already done)
sudo rosdep init 2>/dev/null || warn "rosdep already initialised – skipping"
rosdep update

# Source Jazzy for the rest of this script
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash

info "ROS 2 Jazzy installed and sourced."

# ── Step 1: XFCE Desktop ─────────────────────────────────────────────────────
info "=== Step 1: Installing XFCE desktop ==="
sudo apt-get install -y \
    xfce4 \
    xfce4-goodies \
    xfce4-terminal \
    thunar \
    mousepad \
    xfce4-taskmanager \
    xfce4-screenshooter \
    xfce4-notifyd \
    dbus-x11

# ── Step 2: RDP (xrdp) ───────────────────────────────────────────────────────
info "=== Step 2: Installing xrdp (RDP server) ==="
sudo apt-get install -y xrdp xorgxrdp

# Configure xrdp to start an XFCE session
sudo bash -c 'cat > /etc/xrdp/startwm.sh << "EOF"
#!/bin/sh
unset DBUS_SESSION_BUS_ADDRESS
unset XDG_RUNTIME_DIR
exec startxfce4
EOF'
sudo chmod +x /etc/xrdp/startwm.sh

# Allow xrdp user to access ssl-cert
sudo adduser xrdp ssl-cert 2>/dev/null || true

# Enable and start xrdp service
sudo systemctl enable xrdp
sudo systemctl restart xrdp

info "xrdp is running on port 3389."

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

# ── Step 3: System dependencies (apt) ────────────────────────────────────────
info "=== Step 3/6: Updating apt package lists ==="
sudo apt-get update -y

info "=== Step 4/6: Installing system apt packages ==="
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

# GPIO libraries (lgpio + RPi.GPIO fallback)
info "=== GPIO libraries ==="
apt_install \
    lgpio \
    python3-lgpio 2>/dev/null \
    || warn "lgpio apt package not available – will install via pip"
pip_install \
    lgpio \
    RPi.GPIO

# ── Step 5: ROS 2 packages (apt) ──────────────────────────────────────────────
ROS="ros-${ROS_DISTRO_NAME}"

info "=== Step 5/6: Installing ROS 2 packages ==="

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

# ── Step 6: Python (pip) packages ─────────────────────────────────────────────
info "=== Step 6/6: Installing Python pip packages ==="

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