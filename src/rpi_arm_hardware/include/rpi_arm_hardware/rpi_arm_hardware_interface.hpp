#ifndef RPI_ARM_HARDWARE_INTERFACE_HPP
#define RPI_ARM_HARDWARE_INTERFACE_HPP

#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace rpi_arm_hardware
{

class RPiArmHardwareInterface : public hardware_interface::SystemInterface
{
public:
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams & params) override;

  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

  hardware_interface::return_type write(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

private:
  std::vector<std::string> joint_names_;
  std::vector<double> hw_positions_;
  std::vector<double> hw_commands_;

  std::string host_{"127.0.0.1"};
  int port_{9999};

  int socket_fd_{-1};
  std::string socket_buffer_;

  bool open_socket();
  void close_socket();
  bool write_socket(const std::string & data);
  std::string read_socket_line();
};

}  // namespace rpi_arm_hardware

#endif  // RPI_ARM_HARDWARE_INTERFACE_HPP