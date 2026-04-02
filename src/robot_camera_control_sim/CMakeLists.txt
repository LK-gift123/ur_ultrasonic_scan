cmake_minimum_required(VERSION 3.8)
project(robot_camera_control_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()



# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(PCL REQUIRED)

add_executable(robot_control src/robot_control.cpp)
add_executable(d435i_camera_use src/d435i_camera_use.cpp)
# 新增：碰撞定义节点
add_executable(my_collision_define src/my_collision_define.cpp)
# 新增：PCL 扫描可视化窗口节点
add_executable(pcl_scan_workpiece_window src/pcl_scan_workpiece_window.cpp)

target_include_directories(d435i_camera_use PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(d435i_camera_use
  rclcpp sensor_msgs std_msgs pcl_conversions cv_bridge
)
target_include_directories(robot_control PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(robot_control PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  robot_control
  "rclcpp"
  "geometry_msgs"
  "sensor_msgs"
  "std_msgs"
  "moveit_ros_planning_interface"
  "tf2_geometry_msgs"
  "pcl_conversions"
  "cv_bridge"
  "PCL"
)
ament_target_dependencies(
  my_collision_define
  "rclcpp"
  "moveit_ros_planning_interface"
  "shape_msgs"
  "geometry_msgs"
)

ament_target_dependencies(
  pcl_scan_workpiece_window
  "rclcpp"
  "geometry_msgs"
  "sensor_msgs"
  "pcl_conversions"
  "cv_bridge"
)

target_link_libraries(pcl_scan_workpiece_window ${PCL_LIBRARIES})
target_link_libraries(d435i_camera_use ${PCL_LIBRARIES})

install(TARGETS 
  robot_control
  d435i_camera_use
  my_collision_define
  pcl_scan_workpiece_window
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
