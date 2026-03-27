#include <rclcpp/rclcpp.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>
// 必须引入这个头文件
#include <X11/Xlib.h>
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("pcl_scan_workpiece_window");

    // ==========================================
    // 1. ROS 2 线程（后台运行）
    // 为了将来接收 scan_point 做准备，把 ROS 放后台
    // ==========================================
    std::thread ros_thread([node]() {
        rclcpp::spin(node);
    });

    // ==========================================
    // 2. PCL 渲染器初始化（必须在主线程）
    // ==========================================
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Scanning Viewer"));
    viewer->setBackgroundColor(0.1, 0.1, 0.15); // 深灰色背景，保护眼睛
    viewer->addCoordinateSystem(0.2, "global_frame"); // 添加 20cm 的 XYZ 坐标轴
    
    // 设置相机视角：稍微拉远，从侧上方看着中心点 (0.25, 0.0, 0.05)
    viewer->setCameraPosition(0.8, -0.6, 0.5,   // 相机所在位置
                              0.25, 0.0, 0.05,  // 相机注视的焦点（工件中心）
                              0.0, 0.0, 1.0);   // 相机朝上的方向（Z轴）

    // ==========================================
    // 3. 绘制地面 (参考 my_collision_define.cpp)
    // ==========================================
    // z 的表面在 0.0，厚度 0.05，所以 Z 范围是 -0.05 到 0.0
    viewer->addCube(-1.0, 1.0, -1.0, 1.0, -0.05, 0.0, 0.3, 0.3, 0.3, "ground");
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "ground");

    // ==========================================
    // 4. 绘制工件 Box (严格对齐 URDF 和 碰撞模型)
    // ==========================================
    // 根据 table.xacro: size="0.2 0.2 0.1", origin="0.25 0.0 0.05"
    double x_center = 0.25, y_center = 0.0, z_center = 0.05;
    double dx = 0.1; // 长的一半 (0.2/2)
    double dy = 0.1; // 宽的一半 (0.2/2)
    double dz = 0.05; // 高的一半 (0.1/2)

    double x_min = x_center - dx, x_max = x_center + dx;
    double y_min = y_center - dy, y_max = y_center + dy;
    double z_min = z_center - dz, z_max = z_center + dz;

    // 4.1 绘制半透明实体（木材颜色）
    viewer->addCube(x_min, x_max, y_min, y_max, z_min, z_max, 0.8, 0.5, 0.3, "workpiece_solid");
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.6, "workpiece_solid");

    // 4.2 绘制白色边界线框 (为了看得更清楚)
    viewer->addCube(x_min, x_max, y_min, y_max, z_min, z_max, 1.0, 1.0, 1.0, "workpiece_wireframe");
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, 
                                        pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, 
                                        "workpiece_wireframe");
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2.0, "workpiece_wireframe");

    RCLCPP_INFO(node->get_logger(), "PCL 显示窗口已启动。请使用鼠标左键旋转，滚轮缩放，按住滚轮平移。");

    // ==========================================
    // 5. PCL 主渲染循环
    // ==========================================
    while (!viewer->wasStopped() && rclcpp::ok())
    {
        viewer->spinOnce(50);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // 安全退出
    rclcpp::shutdown();
    ros_thread.join();
    return 0;
}