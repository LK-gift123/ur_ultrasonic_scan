#include <rclcpp/rclcpp.hpp>

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("robot_control_sim");
    RCLCPP_INFO(node->get_logger(), "机器人控制节点预留占位成功！");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}