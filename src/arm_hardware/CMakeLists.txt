cmake_minimum_required(VERSION 3.8)
project(arm_hardware)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(sensor_msgs REQUIRED)

include_directories(include)

add_library(arm_hardware SHARED
  src/arm_hardware_interface.cpp
)

ament_target_dependencies(arm_hardware
  rclcpp
  hardware_interface
  pluginlib
  sensor_msgs
)

pluginlib_export_plugin_description_file(hardware_interface plugin.xml)

install(TARGETS arm_hardware
  DESTINATION lib
)

install(DIRECTORY include/
  DESTINATION include
)

install(PROGRAMS
  scripts/servo_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
