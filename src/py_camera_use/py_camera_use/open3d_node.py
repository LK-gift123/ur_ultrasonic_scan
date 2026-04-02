#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright 2026 [Your Name/Lab]
功能：
1. 纯 Python + Open3D 架构，彻底告别 C++ 内存崩溃。
2. 修复 Numpy 结构化数组 (numpy.void) 转换问题。
3. 包含智能降频、直通、体素、RANSAC、DBSCAN 聚类、法线计算。
4. 真 3D 红色蛇形切片轨迹预览。
5. 首次收到数据时，自动重置 3D 视角居中显示。
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import threading
import time

class Open3DVisionNode(Node):
    def __init__(self):
        super().__init__('open3d_vision_node')
        
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.pc_callback,
            rclpy.qos.qos_profile_sensor_data)

        # 降频保护：每 1.0 秒处理一帧
        self.process_interval = 1.0 
        self.last_time = time.time()

        self.o3d_cloud = o3d.geometry.PointCloud()
        self.sliced_cloud = o3d.geometry.PointCloud()
        self.cloud_lock = threading.Lock()

        self.get_logger().info("Python + Open3D 视觉节点已启动！正在等待点云...")

    def pc_callback(self, msg):
        current_time = time.time()
        if current_time - self.last_time < self.process_interval:
            return
        self.last_time = current_time

        self.get_logger().info("--> 捕获新帧，开始 Open3D 极速处理...")

        # ==========================================
        # 0. ROS PointCloud2 转 Numpy (强制解包修复 numpy.void 问题)
        # ==========================================
        cloud_generator = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        # 【极其关键】：列表推导式提取纯数字，强制转为 Nx3 的 float64 矩阵
        points = np.array([[p[0], p[1], p[2]] for p in cloud_generator], dtype=np.float64)
        
        if points.shape[0] == 0:
            return
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # ==========================================
        # 1. 直通滤波 (PassThrough)
        # ==========================================
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1.0, -1.0, 0.1), max_bound=(1.0, 1.0, 0.8))
        pcd = pcd.crop(bbox)

        # ==========================================
        # 2. 体素降采样
        # ==========================================
        pcd = pcd.voxel_down_sample(voxel_size=0.003)

        # ==========================================
        # 3. RANSAC 桌面剔除
        # ==========================================
        if len(pcd.points) > 50:
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
            pcd = pcd.select_by_index(inliers, invert=True)

        # ==========================================
        # 4. DBSCAN 密度聚类提取目标
        # ==========================================
        if len(pcd.points) > 50:
            labels = np.array(pcd.cluster_dbscan(eps=0.015, min_points=100, print_progress=False))
            max_label = labels.max()
            if max_label >= 0:
                counts = np.bincount(labels[labels >= 0])
                target_label = counts.argmax()
                target_indices = np.where(labels == target_label)[0]
                pcd = pcd.select_by_index(target_indices)

        # ==========================================
        # 5. PCA 表面法线估计
        # ==========================================
        if len(pcd.points) > 0:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
            pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
            # 涂成深灰色
            pcd.paint_uniform_color([0.3, 0.3, 0.3])

        # ==========================================
        # 6. 真 3D 切片预览
        # ==========================================
        slice_pcd = o3d.geometry.PointCloud()
        if len(pcd.points) > 0:
            pts = np.asarray(pcd.points)
            min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
            
            slice_step = 0.045     # 45mm 步距
            slice_thickness = 0.002 # 2mm 切片厚度
            
            sliced_points = []
            for y in np.arange(min_y, max_y, slice_step):
                mask = (pts[:, 1] >= y - slice_thickness) & (pts[:, 1] <= y + slice_thickness)
                sliced_points.append(pts[mask])
                
            if sliced_points:
                all_sliced_pts = np.vstack(sliced_points)
                slice_pcd.points = o3d.utility.Vector3dVector(all_sliced_pts)
                # 轨迹涂成亮红色
                slice_pcd.paint_uniform_color([1.0, 0.0, 0.0])

        self.get_logger().info(f"<-- 处理完成！工件点数: {len(pcd.points)} | 切片轨迹点数: {len(slice_pcd.points)}")

        with self.cloud_lock:
            self.o3d_cloud = pcd
            self.sliced_cloud = slice_pcd


def main(args=None):
    rclpy.init(args=args)
    node = Open3DVisionNode()

    # ROS 2 在后台线程运行
    executor_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    executor_thread.start()

    # Open3D 渲染窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D UR Sensor Slicing Preview", width=800, height=600)
    
    geom_pcd = o3d.geometry.PointCloud()
    geom_slice = o3d.geometry.PointCloud()
    vis.add_geometry(geom_pcd)
    vis.add_geometry(geom_slice)

    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = np.asarray([0.05, 0.05, 0.05])

    # 首次数据标志位，用于自动重置视角
    first_frame = True

    try:
        while rclpy.ok():
            with node.cloud_lock:
                geom_pcd.points = node.o3d_cloud.points
                geom_pcd.colors = node.o3d_cloud.colors
                geom_pcd.normals = node.o3d_cloud.normals
                
                geom_slice.points = node.sliced_cloud.points
                geom_slice.colors = node.sliced_cloud.colors
                
            vis.update_geometry(geom_pcd)
            vis.update_geometry(geom_slice)
            
            # 【自动对焦修复】：一旦点云非空且是第一帧，强制镜头拉远对准工件
            if first_frame and len(geom_pcd.points) > 0:
                vis.reset_view_point(True)
                first_frame = False
                
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.05) 
    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()