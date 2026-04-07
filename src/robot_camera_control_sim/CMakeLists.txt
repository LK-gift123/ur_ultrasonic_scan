cmake_minimum_required(VERSION 3.8)
project(robot_camera_control_sim)

# 强制使用 C++17 标准 (修复 PCL 编译报错的终极杀招)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ==========================================
# 1. 寻找所有必需的 ROS 2 与第三方包
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
# 2. 编译感知层节点 (视觉、点云处理)
# ==========================================
# 2.1 主相机感知节点
add_executable(d435i_camera_sim_use src/d435i_camera_sim_use.cpp)
target_include_directories(d435i_camera_sim_use PUBLIC ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
ament_target_dependencies(d435i_camera_sim_use rclcpp sensor_msgs cv_bridge pcl_conversions)
target_link_libraries(d435i_camera_sim_use ${OpenCV_LIBS} ${PCL_LIBRARIES})

# 2.2 离线 PCD 点云查看器
add_executable(view_pcd_result src/view_pcd_result.cpp)
target_include_directories(view_pcd_result PUBLIC ${PCL_INCLUDE_DIRS})
ament_target_dependencies(view_pcd_result rclcpp)
target_link_libraries(view_pcd_result ${PCL_LIBRARIES})

# 2.3 扫描进度可视化窗口
add_executable(pcl_scan_workpiece_window src/pcl_scan_workpiece_window.cpp)
target_include_directories(pcl_scan_workpiece_window PUBLIC ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
ament_target_dependencies(pcl_scan_workpiece_window rclcpp geometry_msgs sensor_msgs cv_bridge pcl_conversions)
target_link_libraries(pcl_scan_workpiece_window ${OpenCV_LIBS} ${PCL_LIBRARIES})

# ==========================================
# 3. 编译规划与控制层节点 (MoveIt、轨迹)
# ==========================================
# 3.1 碰撞定义节点
add_executable(my_collision_define src/my_collision_define.cpp)
ament_target_dependencies(my_collision_define rclcpp moveit_ros_planning_interface shape_msgs geometry_msgs)

# 3.2 轨迹规划节点 (结合了 PCL 和 ROS)
add_executable(trajectory_planner_sim src/Trajectory_Planner_Sim.cpp)
target_include_directories(trajectory_planner_sim PUBLIC ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(trajectory_planner_sim rclcpp geometry_msgs pcl_conversions)
target_link_libraries(trajectory_planner_sim ${PCL_LIBRARIES})

# 3.3 ⭐ 预留的机器人控制核心节点 (预先绑定 MoveIt 依赖)
add_executable(robot_control_sim src/robot_control_sim.cpp)
ament_target_dependencies(robot_control_sim rclcpp geometry_msgs moveit_ros_planning_interface)

# ==========================================
# 4. 统一安装规则 (清理了冗余的 install 块)
# ==========================================
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# 一次性安装所有 6 个生成的可执行文件
install(TARGETS
  d435i_camera_sim_use
  view_pcd_result
  pcl_scan_workpiece_window
  my_collision_define
  trajectory_planner_sim
  robot_control_sim
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()