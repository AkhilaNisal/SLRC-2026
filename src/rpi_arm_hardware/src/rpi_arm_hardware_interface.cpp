#include "rpi_arm_hardware/rpi_arm_hardware_interface.hpp"
#include "pluginlib/class_list_macros.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>

namespace rpi_arm_hardware
{

hardware_interface::CallbackReturn
RPiArmHardwareInterface::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (SystemInterface::on_init(params) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }

  auto & hw_params = params.hardware_info.hardware_parameters;

  if (hw_params.find("host") != hw_params.end()) {
    host_ = hw_params.at("host");
  }

  if (hw_params.find("port") != hw_params.end()) {
    port_ = std::stoi(hw_params.at("port"));
  }

  const size_t n_joints = params.hardware_info.joints.size();
  hw_positions_.resize(n_joints, 0.0);
  hw_commands_.resize(n_joints, 0.0);
  joint_names_.resize(n_joints);

  for (size_t i = 0; i < n_joints; ++i) {
    joint_names_[i] = params.hardware_info.joints[i].name;
    RCLCPP_INFO(
      rclcpp::get_logger("RPiArmHardware"),
      "Found joint: %s", joint_names_[i].c_str());
  }

  RCLCPP_INFO(
    rclcpp::get_logger("RPiArmHardware"),
    "Initialized %zu joints, bridge at %s:%d",
    n_joints, host_.c_str(), port_);

  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
RPiArmHardwareInterface::on_configure(const rclcpp_lifecycle::State &)
{
  if (!open_socket()) {
    RCLCPP_ERROR(
      rclcpp::get_logger("RPiArmHardware"),
      "Failed to connect to Python bridge at %s:%d",
      host_.c_str(), port_);
    return CallbackReturn::ERROR;
  }

  RCLCPP_INFO(
    rclcpp::get_logger("RPiArmHardware"),
    "Connected to Python bridge at %s:%d",
    host_.c_str(), port_);

  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
RPiArmHardwareInterface::on_activate(const rclcpp_lifecycle::State &)
{
  for (size_t i = 0; i < hw_positions_.size(); ++i) {
    set_state(joint_names_[i] + "/position", hw_positions_[i]);
  }

  RCLCPP_INFO(rclcpp::get_logger("RPiArmHardware"), "RPi hardware activated");
  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
RPiArmHardwareInterface::on_deactivate(const rclcpp_lifecycle::State &)
{
  close_socket();
  RCLCPP_INFO(rclcpp::get_logger("RPiArmHardware"), "RPi hardware deactivated");
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type
RPiArmHardwareInterface::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  if (socket_fd_ < 0) {
    return hardware_interface::return_type::ERROR;
  }

  for (size_t i = 0; i < hw_commands_.size(); ++i) {
    double cmd = get_command(joint_names_[i] + "/position");
    if (std::isnan(cmd)) {
      cmd = hw_positions_[i];
    }
    hw_commands_[i] = cmd;
  }

  std::ostringstream ss;
  ss << "J:";
  for (size_t i = 0; i < hw_commands_.size(); ++i) {
    ss << hw_commands_[i];
    if (i + 1 < hw_commands_.size()) {
      ss << ",";
    }
  }
  ss << "\n";

  if (!write_socket(ss.str())) {
    RCLCPP_WARN(rclcpp::get_logger("RPiArmHardware"), "Failed to write to Python bridge");
    return hardware_interface::return_type::ERROR;
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type
RPiArmHardwareInterface::read(const rclcpp::Time &, const rclcpp::Duration &)
{
  if (socket_fd_ < 0) {
    for (size_t i = 0; i < hw_positions_.size(); ++i) {
      set_state(joint_names_[i] + "/position", hw_commands_[i]);
    }
    return hardware_interface::return_type::OK;
  }

  std::string line = read_socket_line();

  if (!line.empty() && line.rfind("S:", 0) == 0) {
    std::stringstream ss(line.substr(2));
    for (size_t i = 0; i < hw_positions_.size(); ++i) {
      char comma = 0;
      ss >> hw_positions_[i];
      if (i + 1 < hw_positions_.size()) {
        ss >> comma;
      }
      set_state(joint_names_[i] + "/position", hw_positions_[i]);
    }
  } else {
    for (size_t i = 0; i < hw_positions_.size(); ++i) {
      set_state(joint_names_[i] + "/position", hw_commands_[i]);
    }
  }

  return hardware_interface::return_type::OK;
}

bool RPiArmHardwareInterface::open_socket()
{
  socket_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd_ < 0) {
    return false;
  }

  sockaddr_in server_addr{};
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(static_cast<uint16_t>(port_));

  if (::inet_pton(AF_INET, host_.c_str(), &server_addr.sin_addr) <= 0) {
    ::close(socket_fd_);
    socket_fd_ = -1;
    return false;
  }

  if (::connect(socket_fd_, reinterpret_cast<sockaddr *>(&server_addr), sizeof(server_addr)) < 0) {
    ::close(socket_fd_);
    socket_fd_ = -1;
    return false;
  }

  return true;
}

void RPiArmHardwareInterface::close_socket()
{
  if (socket_fd_ >= 0) {
    ::close(socket_fd_);
    socket_fd_ = -1;
  }
}

bool RPiArmHardwareInterface::write_socket(const std::string & data)
{
  if (socket_fd_ < 0) {
    return false;
  }

  const char * ptr = data.c_str();
  size_t total = 0;
  size_t remaining = data.size();

  while (remaining > 0) {
    ssize_t written = ::send(socket_fd_, ptr + total, remaining, 0);
    if (written <= 0) {
      return false;
    }
    total += static_cast<size_t>(written);
    remaining -= static_cast<size_t>(written);
  }

  return true;
}

std::string RPiArmHardwareInterface::read_socket_line()
{
  if (socket_fd_ < 0) {
    return "";
  }

  char buf[256];
  ssize_t n = ::recv(socket_fd_, buf, sizeof(buf), MSG_DONTWAIT);
  if (n <= 0) {
    return "";
  }

  socket_buffer_ += std::string(buf, static_cast<size_t>(n));

  size_t pos = socket_buffer_.find('\n');
  if (pos != std::string::npos) {
    std::string line = socket_buffer_.substr(0, pos);
    socket_buffer_ = socket_buffer_.substr(pos + 1);
    return line;
  }

  return "";
}

}  // namespace rpi_arm_hardware

PLUGINLIB_EXPORT_CLASS(
  rpi_arm_hardware::RPiArmHardwareInterface,
  hardware_interface::SystemInterface)