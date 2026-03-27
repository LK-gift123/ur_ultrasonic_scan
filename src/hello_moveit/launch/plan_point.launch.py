import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # ========================================================================
    # 1. 启动 MoveIt 核心服务及 RViz
    # ========================================================================
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ur7e_moveit_config'),
                         'launch', 'my_moveit_rviz.launch.py')
        )
    )

    # ========================================================================
    # 2. 启动 D435i 真实深度相机节点，并通过传递参数修改内部设置
    # ========================================================================
    # realsense_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(get_package_share_directory('realsense2_camera'),
    #                      'launch', 'rs_launch.py')
    #     ),
    #     launch_arguments={
    #         # ---------------- 【核心开启参数 (本次任务必开)】 ----------------
    #         'enable_color': 'true',         # 开启 RGB 彩色图像流
    #         'enable_depth': 'true',         # 开启红外深度流
    #         'align_depth.enable': 'true',   # 核心：将深度图的像素对齐到彩色图上。如果不开启，生成的点云没有真实颜色。
    #         'pointcloud.enable': 'true',    # 核心：开启后，节点会自动将深度图换算成 3D 点云话题 (/camera/depth/color/points)
    #         'enable_sync': 'true',          # 强制将彩色帧和深度帧的时间戳进行底层同步，方便算法处理
    #         'initial_reset': 'true',        # 启动时硬重置相机 USB 接口。强烈建议开启，防止频繁拔插导致设备占用报错。

    #         # ---------------- 【画质与性能调优参数 (当前激活)】 ----------------
    #         'depth_module.profile': '640x480x30', # 限定深度图为 640x480 分辨率，30 帧。降分辨率可以极大降低电脑 CPU 负担
    #         'rgb_camera.profile': '640x480x30',   # 限定彩色图分辨率
    #         'filters': 'spatial,temporal,hole_filling', # 官方画质增强神器。spatial(空间平滑), temporal(时间降噪), hole_filling(填补黑洞)

    #         # ---------------- 【备用/进阶参数 (已注释，按需解除注释)】 ----------------
    #         # 'camera_name': 'camera',           # 更改相机话题的前缀。如果你用两台 D435i，必须分别改为 camera1, camera2 以防冲突。
    #         # 'serial_no': '',                   # 绑定相机的物理硬件序列号。多台相机同时插在电脑上时，用来指定启动哪一台。
    #         # 'usb_port_id': '',                 # 绑定相机的物理主板 USB 接口编号。
    #         # 'device_type': 'd435i',            # 显式指定寻找 d435i 设备。
    #         # 'clip_distance': '1.5',            # 距离截断（米）。例如填 1.5，则相机只保留 1.5米 以内的点云，远处的全部直接剔除。这对消除杂乱背景极有效！
    #         # 'enable_infra1': 'false',          # 是否输出左红外摄像头的黑白原始画面
    #         # 'enable_infra2': 'false',          # 是否输出右红外摄像头的黑白原始画面
    #         # 'enable_gyro': 'true',             # (D435i专属) 开启内置陀螺仪数据流
    #         # 'enable_accel': 'true',            # (D435i专属) 开启内置加速度计数据流
    #         # 'unite_imu_method': 'linear_interpolation', # (D435i专属) IMU 数据融合方式，通常用于 VSLAM 算法
    #         # 'publish_tf': 'true',              # 是否向 ROS 网络发布相机内部镜头间的静态 TF 坐标变换
    #     }.items()
    # )

    # ========================================================================
    # 3. 启动自定义环境节点
    # ========================================================================
    # 碰撞定义节点
    collision_node = Node(
        package='hello_moveit',
        executable='my_collision_define',
        output='screen',
    )

    # 运动规划节点（通常手动执行，此处可配置是否跟随 launch 一起启动）
    moveit_node = Node(
        package='hello_moveit',
        executable='my_moveit_use',
        output='screen',
    )

    # PCL 实时渲染窗口节点
    pcl_scan_workpiece_node = Node(
        package='hello_moveit',
        executable='pcl_scan_workpiece_window',
        output='screen',
    )

    return LaunchDescription([
        moveit_launch,
        # realsense_launch,         # <-- 成功将深度相机加入启动序列
        collision_node,   
        pcl_scan_workpiece_node,    
    ])