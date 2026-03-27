import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # ========================================================================
    # 启动 D435i 真实深度相机节点，并配置底层参数
    # ========================================================================
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense2_camera'),
                         'launch', 'rs_launch.py')
        ),
        launch_arguments={
            # ---------------- 【核心开启参数 (本次任务必开)】 ----------------
            'enable_color': 'true',         
            'enable_depth': 'true',         
            'align_depth.enable': 'true',   
            'pointcloud.enable': 'true',    
            'enable_sync': 'true',          
            'initial_reset': 'true',        

            # ---------------- 【画质与性能调优参数】 ----------------
            'depth_module.profile': '640x480x30', 
            'rgb_camera.profile': '640x480x30',   
            # 核心开关：必须在这里声明启用了哪些滤波器，下面的高级调参才会生效！
            'filters': 'spatial,temporal,hole_filling', 

            # ========================================================================
            # ---------------- 【深度滤波器高级调参区 (可根据 rqt 调试结果修改)】 ----------------
            # ========================================================================
            
            # 1. 空间滤波器 (Spatial Filter) - 像“磨皮”一样平滑点云表面
            # magnitude: 滤波强度。范围 1~5。值越大，表面越平滑，但工件的锐利边缘会被抹平。提取表面法向量建议调高。
            'spatial_filter.magnitude': '4',
            # smooth_alpha: 平滑权重。范围 0.25~1.0。越小平滑效果越强。
            'spatial_filter.smooth_alpha': '0.5',
            # smooth_delta: 边缘保护阈值。范围 1~50。深度差异大于此值的将被视为物理边缘而不被平滑。
            'spatial_filter.smooth_delta': '20',

            # 2. 时间滤波器 (Temporal Filter) - 消除点云的“跳动”和闪烁噪点
            # alpha: 历史帧权重。范围 0~1。值越小，越相信历史数据，画面越稳定（防闪烁），但移动相机时会有“拖影”。静止扫描工件建议调小。
            'temporal_filter.alpha': '0.2',
            # delta: 运动判定阈值。范围 1~100。如果相邻两帧深度变化超过此值，则认为是物体在动，打断历史融合。
            'temporal_filter.delta': '20',

            # 3. 填洞滤波器 (Hole Filling Filter) - 强行修补黑洞
            # mode: 填补策略。
            #   0 = farest_from_around (用洞周围最远的点填补，最保守)
            #   1 = nearest_from_around (用洞周围最近的点填补)
            #   2 = hole_fill (极其暴力，用极广的范围进行插值填补，不管多大的洞都抹平)
            # 警告：超声探伤任务中，不建议设为 2，否则可能会在空气中凭空造出一块“假曲面”导致机械臂撞击。
            'hole_filling_filter.mode': '1',

            # ---------------- 【备用/进阶参数 (已注释，按需解除注释)】 ----------------
            # 'camera_name': 'camera',           
            # 'clip_distance': '1.5', # 极其好用！直接把 1.5 米外的杂乱背景像切豆腐一样切掉。         
            # 'serial_no': '',                   
            # 'usb_port_id': '',                 
            # 'device_type': 'd435i',            
            # 'enable_gyro': 'true',             
            # 'enable_accel': 'true',            
            # 'publish_tf': 'true',              
        }.items()
    )

    return LaunchDescription([
        realsense_launch
    ])