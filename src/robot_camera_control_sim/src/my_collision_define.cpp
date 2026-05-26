#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <geometry_msgs/msg/pose.h>
// 新增：用于加载和处理 STL 网格的库
#include <geometric_shapes/shapes.h>
#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_operations.h>

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("my_collision_define");
    auto logger = node->get_logger();

    rclcpp::sleep_for(std::chrono::seconds(2));

    moveit::planning_interface::PlanningSceneInterface psi;

    // 创建碰撞对象
    moveit_msgs::msg::CollisionObject custom_workpiece;
    custom_workpiece.header.frame_id = "world";
    custom_workpiece.id = "custom_surface_workpiece"; 

    // =======================================================
    // 🌟 核心修改：从 STL 文件加载网格模型 (Mesh)
    // =======================================================
    Eigen::Vector3d scale(1.0, 1.0, 1.0); // 模型缩放比例，如果 SW 导出的是毫米，这里可能需要改成 0.001
    
    // 注意：确保 package:// 路径与你的包名和文件结构完全一致
    shapes::Mesh* m = shapes::createMeshFromResource("package://artifacts_workpiece_description/meshes/workpiece_base_link.STL", scale);
    
    if (m == nullptr) {
        RCLCPP_ERROR(logger, "❌ 无法加载异形曲面 STL 文件，请检查路径！");
        return 1;
    }

    shape_msgs::msg::Mesh mesh_msg;
    shapes::ShapeMsg shape_msg_base;
    shapes::constructMsgFromShape(m, shape_msg_base);
    mesh_msg = boost::get<shape_msgs::msg::Mesh>(shape_msg_base);

    // 将网格添加到碰撞对象中
    custom_workpiece.meshes.push_back(mesh_msg);

    // 设置在世界中的位置 (需与 Gazebo 里的坐标保持一致)
    geometry_msgs::msg::Pose mesh_pose;
    mesh_pose.orientation.w = 1.0;
    mesh_pose.position.x = 0.0;
    mesh_pose.position.y = 0.0;
    mesh_pose.position.z = 0.82; 
    custom_workpiece.mesh_poses.push_back(mesh_pose);

    custom_workpiece.operation = custom_workpiece.ADD;
    
    // 应用到 MoveIt 规划场景
    psi.applyCollisionObject(custom_workpiece);
    
    RCLCPP_INFO(logger, "✅ 异形曲面 (Mesh) 已成功加入 MoveIt 避障环境！");

    rclcpp::sleep_for(std::chrono::seconds(1));
    rclcpp::shutdown();
    return 0;
}