#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// OpenCV 包含
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

// PCL 包含
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <mutex>
#include <memory>

using std::placeholders::_1;

class CameraSimUseNode : public rclcpp::Node
{
public:
    CameraSimUseNode() : Node("d435i_camera_sim_use")
    {
        RCLCPP_INFO(this->get_logger(), "相机感知节点已启动，等待数据流...");

        // 1. 初始化订阅器 (话题名字必须与 Gazebo 里对齐)
        rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/color/image_raw", 10, std::bind(&CameraSimUseNode::rgb_callback, this, _1));
        
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/depth/image_raw", 10, std::bind(&CameraSimUseNode::depth_callback, this, _1));
        
        pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/camera/camera/depth/color/points", 10, std::bind(&CameraSimUseNode::pc_callback, this, _1));

        // 2. 初始化 PCL 3D 窗口
        viewer_ = std::make_shared<pcl::visualization::PCLVisualizer>("3D Point Cloud Viewer");
        viewer_->setBackgroundColor(0.05, 0.05, 0.05); // 深灰色背景
        viewer_->addCoordinateSystem(0.1);             // 添加坐标轴

        // 3. 初始化 OpenCV 窗口
        cv::namedWindow("RGB Stream", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Depth Stream", cv::WINDOW_AUTOSIZE);

        // 4. 创建定时器，以 30Hz (约 33ms) 的频率刷新 UI，防止界面卡死
        ui_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33), std::bind(&CameraSimUseNode::update_ui, this));
    }

    ~CameraSimUseNode()
    {
        cv::destroyAllWindows();
    }

private:
    // --- 订阅器与数据存储 ---
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;

    cv::Mat current_rgb_;
    cv::Mat current_depth_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_cloud_{new pcl::PointCloud<pcl::PointXYZRGB>};

    std::mutex data_mutex_; // 线程锁

    // --- 可视化组件 ---
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;
    rclcpp::TimerBase::SharedPtr ui_timer_;
    bool is_first_cloud_ = true;

    // --- 回调函数：处理 RGB 彩色图像 ---
    void rgb_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            std::lock_guard<std::mutex> lock(data_mutex_);
            current_rgb_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    // --- 回调函数：处理深度图像 ---
    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            std::lock_guard<std::mutex> lock(data_mutex_);
            cv::Mat depth_raw = cv_bridge::toCvCopy(msg, msg->encoding)->image;
            
            // 深度图一般是 16位 或 32位 浮点数，为了用 imshow 显示，必须归一化到 8位 (0-255)
            cv::normalize(depth_raw, current_depth_, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            cv::applyColorMap(current_depth_, current_depth_, cv::COLORMAP_JET); // 加上伪彩色增强视觉效果
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    // --- 回调函数：处理 3D 点云 ---
    void pc_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        pcl::fromROSMsg(*msg, *current_cloud_);
    }

    // --- 定时器函数：专门用来刷新所有窗口 ---
    void update_ui()
    {
        std::lock_guard<std::mutex> lock(data_mutex_);

        // 刷新 OpenCV 窗口
        if (!current_rgb_.empty()) {
            cv::imshow("RGB Stream", current_rgb_);
        }
        if (!current_depth_.empty()) {
            cv::imshow("Depth Stream", current_depth_);
        }
        cv::waitKey(1); // OpenCV 刷新的关键

        // 刷新 PCL 窗口
        if (!current_cloud_->empty()) {
            if (is_first_cloud_) {
                viewer_->addPointCloud<pcl::PointXYZRGB>(current_cloud_, "cloud");
                viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
                viewer_->resetCamera(); // 第一次收到点云时，自动调整视角
                is_first_cloud_ = false;
            } else {
                viewer_->updatePointCloud<pcl::PointXYZRGB>(current_cloud_, "cloud");
            }
        }
        viewer_->spinOnce(1, true); // PCL 刷新的关键
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraSimUseNode>();
    // 使用默认单线程执行器即可，定时器和回调会交替执行
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}