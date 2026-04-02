cmake_minimum_required(VERSION 3.8)
project(robot_camera_control_sim)

# 强制使用 C++17 标准 (修复 PCL 编译报错的终极杀招)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ==========================================
# 1. 寻找所有必需的包
# ==========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

# 寻找视觉相关库
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)

# ==========================================
# 2. 编译感知层节点：d435i_camera_sim_use
# ==========================================
add_executable(d435i_camera_sim_use src/d435i_camera_sim_use.cpp)
# 告诉编译器 OpenCV 和 PCL 的头文件在哪
target_include_directories(d435i_camera_sim_use PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)
# 链接 ROS 2 的包
ament_target_dependencies(d435i_camera_sim_use
  rclcpp sensor_msgs cv_bridge pcl_conversions
)
# 链接第三方的底层二进制库 (解决 undefine reference 报错的关键)
target_link_libraries(d435i_camera_sim_use
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

# ==========================================
# 3. 编译控制与环境节点
# ==========================================
# 碰撞定义节点
add_executable(my_collision_define src/my_collision_define.cpp)
ament_target_dependencies(my_collision_define
  rclcpp moveit_ros_planning_interface shape_msgs geometry_msgs
)

# 你的 PCL 扫描进度可视化节点 (同样需要 PCL/OpenCV 依赖)
add_executable(pcl_scan_workpiece_window src/pcl_scan_workpiece_window.cpp)
target_include_directories(pcl_scan_workpiece_window PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)
ament_target_dependencies(pcl_scan_workpiece_window
  rclcpp geometry_msgs sensor_msgs cv_bridge pcl_conversions
)
target_link_libraries(pcl_scan_workpiece_window
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

# ==========================================
# 4. 安装规则
# ==========================================
# 安装 launch, urdf, config 等文件夹 (如果有的话)
install(DIRECTORY
  launch
  # urdf
  # config
  DESTINATION share/${PROJECT_NAME}
)

# 安装生成的可执行文件
install(TARGETS
  d435i_camera_sim_use
  my_collision_define
  pcl_scan_workpiece_window
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()