import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory('robot_camera_control_sim')

    # 1. 包含 Gazebo 启动文件
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_dir, 'launch', 'gazebo.launch.py'))
    )

    # 2. 包含 MoveIt & RViz 启动文件
    moveit_rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_dir, 'launch', 'my_moveit_rviz.launch.py'))
    )

    # 3. 包含 PCL 扫描可视化窗口节点 (随系统立即启动)
    pcl_window_node = Node(
        package='robot_camera_control_sim',
        executable='pcl_scan_workpiece_window',
        output='screen'
    )

    # 4. 包含 碰撞定义节点 (⭐ 延时 5 秒启动，等待 MoveIt 完全加载)
    collision_define_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='robot_camera_control_sim',
                executable='my_collision_define',
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        gazebo_launch,
        moveit_rviz_launch,
        pcl_window_node,
        collision_define_node
    ])