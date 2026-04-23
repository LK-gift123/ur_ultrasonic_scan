#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <shape_msgs/msg/mesh.h>
#include <geometry_msgs/msg/pose.h>

// PCL 相关头文件（用于加载 STL 文件）
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <string>

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("my_collision_define");
    auto logger = node->get_logger();

    // 等待 move_group 完全启动
    rclcpp::sleep_for(std::chrono::seconds(2));

    moveit::planning_interface::PlanningSceneInterface psi;

    // -------------------- 地面（与之前相同）--------------------
    moveit_msgs::msg::CollisionObject ground;
    ground.header.frame_id = "world";
    ground.id = "ground";
    shape_msgs::msg::SolidPrimitive ground_primitive;
    ground_primitive.type = ground_primitive.BOX;
    ground_primitive.dimensions = {10.0, 10.0, 0.05};
    ground.primitives.push_back(ground_primitive);
    geometry_msgs::msg::Pose ground_pose;
    ground_pose.orientation.w = 1.0;
    ground_pose.position.z = -0.025;  // 上表面在 z=0
    ground.primitive_poses.push_back(ground_pose);
    ground.operation = ground.ADD;
    psi.applyCollisionObject(ground);
    RCLCPP_INFO(logger, "Ground added.");

    // -------------------- 圆柱体（与之前相同）--------------------
    moveit_msgs::msg::CollisionObject cylinder;
    cylinder.header.frame_id = "world";
    cylinder.id = "cylinder_1";
    shape_msgs::msg::SolidPrimitive cylinder_primitive;
    cylinder_primitive.type = cylinder_primitive.CYLINDER;
    cylinder_primitive.dimensions = {0.10, 0.15};  // 高0.10m，半径0.15m（注意顺序：先高度后半径）
    cylinder.primitives.push_back(cylinder_primitive);
    geometry_msgs::msg::Pose cylinder_pose;
    cylinder_pose.orientation.w = 1.0;
    cylinder_pose.position.x = 0.2;
    cylinder_pose.position.y = 0.0;
    cylinder_pose.position.z = 0.85;
    cylinder.primitive_poses.push_back(cylinder_pose);
    cylinder.operation = cylinder.ADD;
    psi.applyCollisionObject(cylinder);
    RCLCPP_INFO(logger, "Cylinder added.");

    // -------------------- 光学平台（使用 STL 网格）--------------------
    // 1. 加载 STL 文件
    std::string stl_path = "package://optical_platform_description/meshes/table_base_link.STL";
    // 或者使用绝对路径，例如："/home/lk/ur_sim/src/optical_platform_description/meshes/table_base_link.STL"
    // 为了可靠，这里先用绝对路径示范，您可以根据需要改为 package:// 方式（需要解析）
    std::string absolute_path = "/home/lk/ur_sim/src/optical_platform_description/meshes/table_base_link.STL";

    pcl::PolygonMesh mesh;
    if (pcl::io::loadPolygonFileSTL(absolute_path, mesh) == -1)
    {
        RCLCPP_ERROR(logger, "Failed to load STL file: %s", absolute_path.c_str());
        rclcpp::shutdown();
        return 1;
    }

    // 2. 将 PCL 的 PolygonMesh 转换为 shape_msgs::msg::Mesh
    shape_msgs::msg::Mesh mesh_msg;
    // 提取顶点
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromPCLPointCloud2(mesh.cloud, cloud);
    mesh_msg.vertices.resize(cloud.points.size());
    for (size_t i = 0; i < cloud.points.size(); ++i)
    {
        mesh_msg.vertices[i].x = cloud.points[i].x;
        mesh_msg.vertices[i].y = cloud.points[i].y;
        mesh_msg.vertices[i].z = cloud.points[i].z;
    }
    // 提取三角形面片索引
    mesh_msg.triangles.resize(mesh.polygons.size());
    for (size_t i = 0; i < mesh.polygons.size(); ++i)
    {
        const auto& poly = mesh.polygons[i];
        if (poly.vertices.size() != 3)
        {
            RCLCPP_WARN(logger, "Non-triangle polygon ignored.");
            continue;
        }
        mesh_msg.triangles[i].vertex_indices[0] = poly.vertices[0];
        mesh_msg.triangles[i].vertex_indices[1] = poly.vertices[1];
        mesh_msg.triangles[i].vertex_indices[2] = poly.vertices[2];
    }

    // 3. 创建碰撞对象
    moveit_msgs::msg::CollisionObject platform;
    platform.header.frame_id = "world";
    platform.id = "optical_platform";
    platform.meshes.push_back(mesh_msg);

    // 设置位姿：中心位于 (0, 0, 0.8)，无旋转（根据您的需求）
    geometry_msgs::msg::Pose platform_pose;
    platform_pose.orientation.w = 1.0;
    platform_pose.position.x = 0.0;
    platform_pose.position.y = 0.0;
    platform_pose.position.z = 0.8;
    platform.mesh_poses.push_back(platform_pose);

    platform.operation = platform.ADD;
    psi.applyCollisionObject(platform);
    RCLCPP_INFO(logger, "Optical platform added at (0,0,0.8).");

    // 保持节点运行一小段时间，确保消息被处理
    rclcpp::sleep_for(std::chrono::seconds(1));
    rclcpp::shutdown();
    return 0;
}