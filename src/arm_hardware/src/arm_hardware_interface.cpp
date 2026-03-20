#include "arm_hardware/arm_hardware_interface.hpp"
#include "pluginlib/class_list_macros.hpp"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace arm_hardware
{

hardware_interface::CallbackReturn
ArmHardwareInterface::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (SystemInterface::on_init(params) != CallbackReturn::SUCCESS)
    return CallbackReturn::ERROR;

  size_t n = params.hardware_info.joints.size();

  hw_positions_.resize(n, 0.0);
  hw_commands_.resize(n, 0.0);
  joint_names_.resize(n);

  for (size_t i = 0; i < n; i++)
  {
    joint_names_[i] = params.hardware_info.joints[i].name;
  }

  // ROS node inside hardware plugin
  node_ = rclcpp::Node::make_shared("arm_hardware_node");

  pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(
    "/joint_commands", 10);

  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
ArmHardwareInterface::on_activate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(node_->get_logger(), "Hardware Activated");
  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
ArmHardwareInterface::on_deactivate(const rclcpp_lifecycle::State &)
{
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type
ArmHardwareInterface::read(const rclcpp::Time &, const rclcpp::Duration &)
{
  for (size_t i = 0; i < hw_positions_.size(); i++)
  {
    hw_positions_[i] = hw_commands_[i];
    set_state(joint_names_[i] + "/position", hw_positions_[i]);
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type
ArmHardwareInterface::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  sensor_msgs::msg::JointState msg;
  msg.name = joint_names_;
  msg.position.resize(hw_commands_.size());

  for (size_t i = 0; i < hw_commands_.size(); i++)
  {
    double cmd = get_command(joint_names_[i] + "/position");

    if (std::isnan(cmd))
      cmd = hw_positions_[i];

    hw_commands_[i] = cmd;
    msg.position[i] = cmd;
  }

  pub_->publish(msg);

  return hardware_interface::return_type::OK;
}

} // namespace arm_hardware

PLUGINLIB_EXPORT_CLASS(
  arm_hardware::ArmHardwareInterface,
  hardware_interface::SystemInterface)