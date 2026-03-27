/*
 * Copyright 2026 [Your Name/Lab]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * ... (Apache 2.0 License) ...
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>

// PCL 相关
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h>

// OpenCV 与 ROS 图像桥接
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <mutex>
#include <X11/Xlib.h>

class CameraUseNode : public rclcpp::Node
{
public:
    // ==========================================
    // 🎛️ 可视化功能控制面板 (按需修改 true / false)
    // ==========================================
    bool enable_3d_pointcloud_ = true; // 是否开启 3D 点云 PCL 窗口
    bool enable_2d_color_img_  = true; // 是否开启 2D 原生彩色 OpenCV 窗口
    bool enable_2d_depth_img_  = true; // 是否开启 2D 深度伪彩 OpenCV 窗口

    CameraUseNode() : Node("my_camera_use")
    {
        // 1. 初始化 3D 点云订阅
        if (enable_3d_pointcloud_) {
            pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/camera/camera/depth/color/points", rclcpp::SensorDataQoS(),
                std::bind(&CameraUseNode::pointcloud_callback, this, std::placeholders::_1));
        }

        // 2. 初始化 2D 彩色图像订阅
        if (enable_2d_color_img_) {
            color_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                "/camera/camera/color/image_raw", 10,
                std::bind(&CameraUseNode::color_img_callback, this, std::placeholders::_1));
        }

        // 3. 初始化 2D 深度图像订阅 (注意：使用 align对齐后的深度图，与彩色图视野一致)
        if (enable_2d_depth_img_) {
            depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                "/camera/camera/aligned_depth_to_color/image_raw", 10,
                std::bind(&CameraUseNode::depth_img_callback, this, std::placeholders::_1));
        }

        RCLCPP_INFO(this->get_logger(), "视觉接收与可视化节点已启动！");
    }

    // 提供给主循环读取最新数据的接口
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr getLatestCloud() {
        std::lock_guard<std::mutex> lock(pc_mutex_);
        return latest_cloud_;
    }
    cv::Mat getLatestColorImg() {
        std::lock_guard<std::mutex> lock(color_mutex_);
        return latest_color_img_.clone();
    }
    cv::Mat getLatestDepthImg() {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        return latest_depth_img_.clone();
    }

private:
    // --- 互斥锁与数据缓存 ---
    std::mutex pc_mutex_, color_mutex_, depth_mutex_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr latest_cloud_;
    cv::Mat latest_color_img_;
    cv::Mat latest_depth_img_;

    // --- 订阅器 ---
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;

    // ==========================================
    // 回调函数区：负责接收并转换数据
    // ==========================================
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msg, *cloud);
        
        std::lock_guard<std::mutex> lock(pc_mutex_);
        latest_cloud_ = cloud;
    }

    void color_img_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            // 将 ROS 图像转为 OpenCV Mat (bgr8 格式)
            cv::Mat cv_img = cv_bridge::toCvCopy(msg, "bgr8")->image;
            std::lock_guard<std::mutex> lock(color_mutex_);
            latest_color_img_ = cv_img;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "彩色图像转换失败: %s", e.what());
        }
    }

    void depth_img_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            // 深度图原生是 16位无符号整型 (16UC1, 单位毫米)
            cv::Mat depth_16u = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
            
            // 为了肉眼能看清，必须将其归一化并转为 8 位伪彩色图 (Colormap)
            cv::Mat depth_8u, depth_color;
            // 假设最远有效距离为 2000mm (2米)，进行动态范围缩放
            depth_16u.convertTo(depth_8u, CV_8UC1, 255.0 / 2000.0); 
            cv::applyColorMap(depth_8u, depth_color, cv::COLORMAP_JET); // 涂上 JET 伪彩色（红近蓝远）

            std::lock_guard<std::mutex> lock(depth_mutex_);
            latest_depth_img_ = depth_color;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "深度图像转换失败: %s", e.what());
        }
    }
};

// ==========================================
// 主函数与渲染线程
// ==========================================
int main(int argc, char * argv[])
{
    // 防止 PCL 与 ROS 多线程环境产生 X11 冲突闪退
    XInitThreads();
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraUseNode>();

    // 1. 初始化 PCL 3D 窗口
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    if (node->enable_3d_pointcloud_) {
        viewer = std::make_shared<pcl::visualization::PCLVisualizer>("D435i 3D PointCloud Viewer");
        viewer->setBackgroundColor(0.05, 0.05, 0.05); // 深灰色背景
        viewer->addCoordinateSystem(0.2);
        viewer->initCameraParameters();
    }

    // 2. 初始化 OpenCV 2D 窗口
    if (node->enable_2d_color_img_) cv::namedWindow("2D Color Stream", cv::WINDOW_AUTOSIZE);
    if (node->enable_2d_depth_img_) cv::namedWindow("2D Depth Stream (Colormap)", cv::WINDOW_AUTOSIZE);

    rclcpp::WallRate loop_rate(30); // 维持 30Hz 刷新率

    // 主渲染循环
    while (rclcpp::ok())
    {
        // 让 ROS 节点处理一次回调队列（接收最新数据）
        rclcpp::spin_some(node);

        // --- 更新 3D 点云画面 ---
        if (node->enable_3d_pointcloud_ && viewer && !viewer->wasStopped()) {
            auto cloud = node->getLatestCloud();
            if (cloud && !cloud->empty()) {
                if (!viewer->updatePointCloud(cloud, "d435i_cloud")) {
                    viewer->addPointCloud(cloud, "d435i_cloud");
                    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "d435i_cloud");
                }
            }
            viewer->spinOnce(10);
        }

        // --- 更新 2D OpenCV 画面 ---
        if (node->enable_2d_color_img_) {
            cv::Mat color_img = node->getLatestColorImg();
            if (!color_img.empty()) cv::imshow("2D Color Stream", color_img);
        }

        if (node->enable_2d_depth_img_) {
            cv::Mat depth_img = node->getLatestDepthImg();
            if (!depth_img.empty()) cv::imshow("2D Depth Stream (Colormap)", depth_img);
        }

        // 必须保留 cv::waitKey 刷新 OpenCV 窗口事件
        if (node->enable_2d_color_img_ || node->enable_2d_depth_img_) {
            cv::waitKey(10);
        }

        // 如果用户关掉了 PCL 窗口，则安全退出节点
        if (viewer && viewer->wasStopped()) break; 

        loop_rate.sleep();
    }

    rclcpp::shutdown();
    cv::destroyAllWindows();
    return 0;
}