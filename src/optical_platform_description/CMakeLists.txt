cmake_minimum_required(VERSION 3.8)
project(optical_platform_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(robot_state_publisher REQUIRED)
find_package(gazebo_ros REQUIRED)

# 安装目录
install(
  DIRECTORY meshes urdf
  DESTINATION share/${PROJECT_NAME}
)

# 可选：如果将来添加launch文件
# install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()