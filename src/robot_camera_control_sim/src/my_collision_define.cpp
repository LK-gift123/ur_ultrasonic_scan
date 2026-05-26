#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <geometry_msgs/msg/pose.h>
#include <geometric_shapes/shapes.h>
#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_operations.h>
#include <vector>

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("my_collision_define");
    auto logger = node->get_logger();

    // 延时等待，确保 move_group 节点完全拉起
    rclcpp::sleep_for(std::chrono::seconds(2));

    moveit::planning_interface::PlanningSceneInterface psi;
    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;

    // 模型统一缩放比例 (若导出格式为米，保持 1.0；若为毫米，需改为 0.001)
    Eigen::Vector3d scale(1.0, 1.0, 1.0); 

    // =======================================================
    // 🌟 1. 导入原有物体：异形曲面 (Mesh)
    // =======================================================
    moveit_msgs::msg::CollisionObject custom_workpiece;
    custom_workpiece.header.frame_id = "world";
    custom_workpiece.id = "custom_surface_workpiece";

    shapes::Mesh* m_workpiece = shapes::createMeshFromResource(
        "package://artifacts_workpiece_description/meshes/workpiece_base_link.STL", scale);
    
    if (m_workpiece != nullptr) {
        shape_msgs::msg::Mesh mesh_msg;
        shapes::ShapeMsg shape_msg_base;
        shapes::constructMsgFromShape(m_workpiece, shape_msg_base);
        mesh_msg = boost::get<shape_msgs::msg::Mesh>(shape_msg_base);
        
        custom_workpiece.meshes.push_back(mesh_msg);
        geometry_msgs::msg::Pose wp_pose;
        wp_pose.orientation.w = 1.0;
        wp_pose.position.x = 0.0;
        wp_pose.position.y = 0.0;
        wp_pose.position.z = 0.82; // 对应仿真中的绝对高度
        custom_workpiece.mesh_poses.push_back(wp_pose);
        custom_workpiece.operation = custom_workpiece.ADD;
        collision_objects.push_back(custom_workpiece);
        RCLCPP_INFO(logger, "网格模型导入：[异形曲面] 准备就绪");
    } else {
        RCLCPP_ERROR(logger, "❌ 无法加载异形曲面 STL 文件，请检查路径！");
    }

    // =======================================================
    // 🌟 2. 补全环境物体：光学平台 (Optical Platform)
    // =======================================================
    moveit_msgs::msg::CollisionObject optical_platform;
    optical_platform.header.frame_id = "world";
    optical_platform.id = "optical_platform";

    // 自动检索你的物理硬盘 STL 资源文件
    shapes::Mesh* m_platform = shapes::createMeshFromResource(
        "package://optical_platiform_description/meshes/optical_platiform_base_link.STL", scale);

    if (m_platform != nullptr) {
        shape_msgs::msg::Mesh mesh_msg;
        shapes::ShapeMsg shape_msg_base;
        shapes::constructMsgFromShape(m_platform, shape_msg_base);
        mesh_msg = boost::get<shape_msgs::msg::Mesh>(shape_msg_base);

        optical_platform.meshes.push_back(mesh_msg);
        geometry_msgs::msg::Pose platform_pose;
        platform_pose.orientation.w = 1.0;
        // 对齐你在 gazebo.launch.py 和 ur7e.urdf.xacro 中配置的 table_to_world 的物理原点
        platform_pose.position.x = 0.0;
        platform_pose.position.y = 0.0;
        platform_pose.position.z = 0.8; 
        
        optical_platform.mesh_poses.push_back(platform_pose);
        optical_platform.operation = optical_platform.ADD;
        collision_objects.push_back(optical_platform);
        RCLCPP_INFO(logger, "✅ 网格模型导入：[光学平台] 成功构建");
    } else {
        RCLCPP_ERROR(logger, "❌ 无法加载光学平台 STL 文件，请检查路径！");
    }

    // =======================================================
    // 🌟 3. 补全环境物体：机械臂底座 (Robot Arm Base Link)
    // =======================================================
    moveit_msgs::msg::CollisionObject robot_base_fixture;
    robot_base_fixture.header.frame_id = "world";
    robot_base_fixture.id = "robot_arm_base_link_fixture";

    // 加载与光学平台组装连接的机器人基础底座模型
    shapes::Mesh* m_base = shapes::createMeshFromResource(
        "package://robot_arm_base_link/meshes/robot_arm_base_link.STL", scale);

    if (m_base != nullptr) {
        shape_msgs::msg::Mesh mesh_msg;
        shapes::ShapeMsg shape_msg_base;
        shapes::constructMsgFromShape(m_base, shape_msg_base);
        mesh_msg = boost::get<shape_msgs::msg::Mesh>(shape_msg_base);

        robot_base_fixture.meshes.push_back(mesh_msg);
        geometry_msgs::msg::Pose base_pose;
        base_pose.orientation.w = 1.0;
        // 对齐你在 ur7e.urdf.xacro 中配置的 base_to_table 的相对于光学平台的相对位移量
        // 光学平台在 0.8，相对于平台坐标原点偏置 (-0.325, -0.625, 0.0)，所以在世界坐标系下高度为 0.8
        base_pose.position.x = -0.325;
        base_pose.position.y = -0.625;
        base_pose.position.z = 0.8;

        robot_base_fixture.mesh_poses.push_back(base_pose);
        robot_base_fixture.operation = robot_base_fixture.ADD;
        collision_objects.push_back(robot_base_fixture);
        RCLCPP_INFO(logger, "✅ 网格模型导入：[机械臂连接底座] 成功构建");
    } else {
        RCLCPP_ERROR(logger, "❌ 无法加载机械臂底座固定夹具的 STL 文件，请检查路径！");
    }

    // =======================================================
    // 🌟 4. 批量异步提交至 MoveIt 规划场景中
    // =======================================================
    if (!collision_objects.empty()) {
        psi.applyCollisionObjects(collision_objects);
        RCLCPP_INFO(logger, "🚀 全局避障环境更新完毕！异形曲面、光学平台与机械臂底座已成功合闸。");
    }

    rclcpp::sleep_for(std::chrono::seconds(1));
    rclcpp::shutdown();
    return 0;
}