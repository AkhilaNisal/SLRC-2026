#pragma once

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/macros.hpp"

#include "sensor_msgs/msg/joint_state.hpp"   // ✅ ADD THIS

#include <vector>
#include <string>

namespace arm_hardware
{

class ArmHardwareInterface : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(ArmHardwareInterface)

  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams & params) override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State &) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State &) override;

  hardware_interface::return_type read(
    const rclcpp::Time &, const rclcpp::Duration &) override;

  hardware_interface::return_type write(
    const rclcpp::Time &, const rclcpp::Duration &) override;

private:
  std::vector<double> hw_positions_;
  std::vector<double> hw_commands_;
  std::vector<std::string> joint_names_;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
};

}