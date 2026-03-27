/*
 * Copyright 2026 [Your Name/Lab]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * * 本代码实现的功能：
 * 1. 作为一个 ROS 2 节点，实时订阅 Intel RealSense D435i 的点云和图像流。
 * 2. 使用 PCL 库对点云进行双重滤波（直通空间裁剪 + 体素降采样），提取核心工件区域。
 * 3. 解决 ROS 2 接收线程与主渲染线程之间的数据竞争问题。
 * 4. 并发使用 PCLVisualizer (3D) 和 OpenCV (2D) 实时渲染传感器数据。
 */

// ==================== 【头文件包含区】 ====================
// ROS 2 核心库及传感器消息类型
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>

// PCL (点云库) 相关头文件
#include <pcl_conversions/pcl_conversions.h> // 用于将 ROS 消息无损转换为 PCL 对象
#include <pcl/visualization/pcl_visualizer.h> // PCL 的 3D 渲染引擎
#include <pcl/filters/voxel_grid.h>          // 体素降采样滤波器 (数据压缩)
#include <pcl/filters/passthrough.h>         // 直通滤波器 (空间裁剪)

// OpenCV (计算机视觉库) 相关头文件
#include <cv_bridge/cv_bridge.h>             // 用于将 ROS 图像消息转换为 OpenCV 的 Mat 矩阵
#include <opencv2/opencv.hpp>                // OpenCV 核心及图形界面功能

// C++ 标准库及系统级组件
#include <mutex>                             // 互斥锁，用于多线程数据保护
#include <X11/Xlib.h>                        // Linux X11 视窗系统底层库，防止图形界面多线程崩溃

// ==================== 【节点类定义】 ====================
class CameraUseNode : public rclcpp::Node
{
public:
    // ---------------------------------------------------------
    // 🎛️ 功能控制面板 (全局配置参数)
    // ---------------------------------------------------------
    // 界面显示开关
    bool enable_3d_pointcloud_ = true; // 是否开启 3D 点云 PCL 窗口
    bool enable_2d_color_img_  = true; // 是否开启 2D 原生彩色 OpenCV 窗口
    bool enable_2d_depth_img_  = true; // 是否开启 2D 深度伪彩 OpenCV 窗口

    // 直通滤波 (PassThrough) 参数：用于像切豆腐一样切除工件周围的杂物和远处的墙壁
    bool enable_passthrough_filter_ = true; 
    float pass_z_min_ = 0.1f;               // 最小保留距离 (米)：裁掉极近处的幽灵点盲区
    float pass_z_max_ = 0.4f;               // 最大保留距离 (米)：1.5表示裁掉 1.5 米外的背景墙壁

    // 体素降采样 (VoxelGrid) 参数：用于降低点云密度，提高后续机械臂法向量计算的速度
    bool enable_voxel_filter_  = true;      
    float voxel_size_ = 0.002f;              // 体素叶子大小：0.01 代表用 1cm x 1cm x 1cm 的方块去合并原始点云

    // 构造函数：节点初始化时执行
    CameraUseNode() : Node("my_camera_use")
    {
        // 初始化订阅器 (Subscription)
        // 使用 std::bind 将类的成员函数作为回调函数绑定到特定的 ROS 话题上
        
        if (enable_3d_pointcloud_) {
            // 订阅对齐后的 3D 彩色点云。注意使用 SensorDataQoS()，这是一种专为高频传感器设计的尽力而为传输策略
            pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/camera/camera/depth/color/points", rclcpp::SensorDataQoS(),
                std::bind(&CameraUseNode::pointcloud_callback, this, std::placeholders::_1));
        }

        if (enable_2d_color_img_) {
            // 订阅彩色 2D 图像
            color_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                "/camera/camera/color/image_raw", 10,
                std::bind(&CameraUseNode::color_img_callback, this, std::placeholders::_1));
        }

        if (enable_2d_depth_img_) {
            // 订阅对齐到彩色视角的深度 2D 图像
            depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                "/camera/camera/aligned_depth_to_color/image_raw", 10,
                std::bind(&CameraUseNode::depth_img_callback, this, std::placeholders::_1));
        }

        RCLCPP_INFO(this->get_logger(), "视觉接收与滤波节点已启动！正在应用直通与体素滤波...");
    }

    // ---------------------------------------------------------
    // 🔒 线程安全的数据读取接口 (供主函数的 while 循环调用)
    // 解释：ROS 2 的回调函数是在后台单独的线程中运行的，而画图渲染在主线程。
    // 如果主线程正在画点云，后台线程突然收到了新点云并覆盖了内存，程序就会段错误崩溃 (Segment Fault)。
    // 所以必须用 lock_guard (互斥锁) 把读取过程锁起来，保证“读的时候不准写”。
    // ---------------------------------------------------------
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr getLatestCloud() {
        std::lock_guard<std::mutex> lock(pc_mutex_);
        return latest_cloud_; // 返回当前缓存的最新点云智能指针
    }
    cv::Mat getLatestColorImg() {
        std::lock_guard<std::mutex> lock(color_mutex_);
        return latest_color_img_.clone(); // 深拷贝一份图像矩阵发出去，防止内存被污染
    }
    cv::Mat getLatestDepthImg() {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        return latest_depth_img_.clone();
    }

private:
    // --- 互斥锁与数据缓存变量 ---
    std::mutex pc_mutex_, color_mutex_, depth_mutex_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr latest_cloud_; // 存放处理完毕的最终点云
    cv::Mat latest_color_img_;                            // 存放当前彩色帧
    cv::Mat latest_depth_img_;                            // 存放当前深度伪彩帧

    // --- ROS 订阅器智能指针 ---
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;

    // ==========================================
    // 📥 回调函数区：传感器数据到达时自动触发
    // ==========================================
    
    // 【核心】点云处理流水线
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // 1. 初始化智能指针并转换格式
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msg, *raw_cloud); // ROS PointCloud2 -> PCL PointCloud
        
        // 准备中间容器
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pass_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr processed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        // 2. 第一道工序：直通滤波 (PassThrough)
        if (enable_passthrough_filter_) {
            pcl::PassThrough<pcl::PointXYZRGB> pass;
            pass.setInputCloud(raw_cloud);             // 喂入原始点云
            pass.setFilterFieldName("z");              // 指定沿摄像头的正前方 (Z轴) 进行刀切
            pass.setFilterLimits(pass_z_min_, pass_z_max_); // 刀切的范围：0.1米 到 1.5米
            pass.filter(*pass_cloud);                  // 执行切割，结果存入 pass_cloud
        } else {
            pass_cloud = raw_cloud; // 不切的话，直接传递给下一道工序
        }

        // 3. 第二道工序：体素降采样 (VoxelGrid)
        if (enable_voxel_filter_) {
            pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
            voxel_filter.setInputCloud(pass_cloud);    // 喂入刚刚直通滤波切好的点云
            voxel_filter.setLeafSize(voxel_size_, voxel_size_, voxel_size_); // 设置 X,Y,Z 三个维度的方块大小
            voxel_filter.filter(*processed_cloud);     // 执行降采样合并，结果存入 processed_cloud

            // 日志节流打印：每 2000 毫秒(2秒)打印一次处理前后的数据量对比，防止终端刷屏卡死
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "双重滤波效果 -> 原始: %zu | 裁剪后: %zu | 降采样后: %zu", 
                raw_cloud->points.size(), pass_cloud->points.size(), processed_cloud->points.size());
        } else {
            processed_cloud = pass_cloud;
        }
        
        // 4. 将纯净的最终点云安全地更新到类的缓存变量中，供渲染线程自取
        std::lock_guard<std::mutex> lock(pc_mutex_); // 抢占锁
        latest_cloud_ = processed_cloud;             // 更新数据
        // 函数结束，lock 生命周期结束，自动释放锁
    }

    // 彩色图像处理
    void color_img_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            // 利用 cv_bridge 将 ROS 独有的图像流格式转换为 OpenCV 通用的 Mat 格式 (BGR8通道)
            cv::Mat cv_img = cv_bridge::toCvCopy(msg, "bgr8")->image;
            std::lock_guard<std::mutex> lock(color_mutex_);
            latest_color_img_ = cv_img;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "彩色图像转换失败: %s", e.what());
        }
    }

    // 深度图像处理
    void depth_img_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            // 深度图原生是 16位无符号整型 (16UC1)，里面存的是毫米为单位的物理距离，人眼看着是一片漆黑的
            cv::Mat depth_16u = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
            
            cv::Mat depth_8u, depth_color;
            // 归一化操作：把 0~2000 毫米的范围，按比例硬压成 0~255 的 8位 像素亮度灰度图
            depth_16u.convertTo(depth_8u, CV_8UC1, 255.0 / 2000.0); 
            // 伪彩色映射：给灰度图涂上 JET 风格的假颜色（红色代表距离近，蓝色代表距离远）
            cv::applyColorMap(depth_8u, depth_color, cv::COLORMAP_JET);

            std::lock_guard<std::mutex> lock(depth_mutex_);
            latest_depth_img_ = depth_color;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "深度图像转换失败: %s", e.what());
        }
    }
};

// ==========================================
// 🚀 主函数：程序的入口与主渲染循环
// ==========================================
int main(int argc, char * argv[])
{
    // [极其关键] 告诉 Linux 视窗系统，我要在多线程里画图了。如果不加这句，PCL 窗口可能会随机闪退
    XInitThreads(); 
    
    rclcpp::init(argc, argv);
    // 实例化刚刚定义的节点对象
    auto node = std::make_shared<CameraUseNode>();

    // ---------------------------------------------------------
    // 初始化可视化窗口
    // ---------------------------------------------------------
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    if (node->enable_3d_pointcloud_) {
        // 创建 3D 查看器
        viewer = std::make_shared<pcl::visualization::PCLVisualizer>("D435i Real-time 3D Viewer");
        viewer->setBackgroundColor(0.05, 0.05, 0.05); // 设置深灰色护眼背景
        viewer->addCoordinateSystem(0.2);             // 在原点画一个 0.2米 长的小坐标系(RGB -> XYZ)
        viewer->initCameraParameters();               // 初始化渲染虚拟镜头的视角
    }

    if (node->enable_2d_color_img_) cv::namedWindow("2D Color Stream", cv::WINDOW_AUTOSIZE);
    if (node->enable_2d_depth_img_) cv::namedWindow("2D Depth Stream (Colormap)", cv::WINDOW_AUTOSIZE);

    // 设置循环频率为 30Hz，正好匹配相机的 30帧 输出速率
    rclcpp::WallRate loop_rate(30);

    // ==========================================
    // 🔁 主渲染循环：只要 ROS 没收到 Ctrl+C，就一直转
    // ==========================================
    while (rclcpp::ok())
    {
        // [核心] rclcpp::spin_some 的作用：
        // 暂停一下主循环，去后台看看传感器有没有发新数据过来。如果有，就去触发那三个 callback 函数。
        // 处理完一次队列后，立刻返回主循环继续往下走，防止整个程序卡死在等待数据里。
        rclcpp::spin_some(node);

        // --- 刷新 3D 点云画面 ---
        if (node->enable_3d_pointcloud_ && viewer && !viewer->wasStopped()) {
            auto cloud = node->getLatestCloud(); // 调用类接口，安全拿出现存最新的点云
            if (cloud && !cloud->empty()) {
                // 如果是第一次，就添加点云；如果之前画过了，就只更新里面的坐标数据，这样性能最高
                if (!viewer->updatePointCloud(cloud, "d435i_cloud")) {
                    viewer->addPointCloud(cloud, "d435i_cloud");
                    // 因为我们前面做了体素降采样，点变少了，所以这里把显示的点画大一点 (Size=3)，看着更连贯
                    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "d435i_cloud"); 
                }
            }
            viewer->spinOnce(10); // 告诉 PCL 去刷新一下屏幕硬件，给 10 毫秒的时间
        }

        // --- 刷新 2D OpenCV 画面 ---
        if (node->enable_2d_color_img_) {
            cv::Mat color_img = node->getLatestColorImg();
            if (!color_img.empty()) cv::imshow("2D Color Stream", color_img); // 把图片贴到窗口上
        }

        if (node->enable_2d_depth_img_) {
            cv::Mat depth_img = node->getLatestDepthImg();
            if (!depth_img.empty()) cv::imshow("2D Depth Stream (Colormap)", depth_img);
        }

        // OpenCV 专属机制：必须调用 waitKey，系统底层才会真正把像素推送到显示器上，同时它也能监听键盘事件
        if (node->enable_2d_color_img_ || node->enable_2d_depth_img_) {
            cv::waitKey(10); 
        }

        // 如果用户点击了 PCL 窗口的红叉关掉了查看器，主动退出 while 循环结束程序
        if (viewer && viewer->wasStopped()) break; 

        // 强行休眠一小会儿，凑够 1/30 秒，防止把 CPU 100% 榨干
        loop_rate.sleep();
    }

    // 优雅地释放系统资源并退出
    rclcpp::shutdown();
    cv::destroyAllWindows();
    return 0;
}