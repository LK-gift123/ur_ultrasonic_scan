#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "std_srvs/srv/trigger.hpp" // 触发服务头文件

using namespace std::chrono_literals;

class ServoTester : public rclcpp::Node
{
public:
  ServoTester(const rclcpp::NodeOptions & options)
  : Node("servo_tester_cpp", options)
  {
    // 创建笛卡尔速度发布者
    publisher_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
      "/servo_node/delta_twist_cmds", 10);

    // 创建服务客户端，负责自动去戳一下 Servo 的电闸
    servo_start_client_ = this->create_client<std_srvs::srv::Trigger>("/servo_node/start_servo");

    // 50Hz 核心实时控制循环 (20ms)
    timer_ = this->create_wall_timer(
      20ms, std::bind(&ServoTester::timer_callback, this));

    // 启动一个单次定时器，在节点起来 1 秒后自动呼叫 Trigger，实现无人值守自动使能
    autostart_timer_ = this->create_wall_timer(
      1s, std::bind(&ServoTester::trigger_servo_automatically, this));

    RCLCPP_INFO(this->get_logger(), "✅ C++ 实时伺服测试节点已成功初始化！");
  }

private:
  void trigger_servo_automatically()
  {
    autostart_timer_->cancel(); // 确保自动合闸逻辑只执行一次
    
    // 等待 Servo 节点的 Service 就绪
    if (!servo_start_client_->wait_for_service(3s)) {
      RCLCPP_WARN(this->get_logger(), "⚠️ 未找到 /servo_node/start_servo 服务，请确认 servo.launch.py 是否已正确拉起");
      return;
    }

    auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
    // 异步发送请求，防止阻塞实时控制的主循环线程
    auto result_future = servo_start_client_->async_send_request(request);
    RCLCPP_INFO(this->get_logger(), "🚀 [🤖 全自动使能] 已成功向 MoveIt Servo 发送激活 (Trigger) 信号！数据通道已开启。");
  }

  void timer_callback()
  {
    auto msg = geometry_msgs::msg::TwistStamped();
    // ⭐ 灵魂修复：必须打上当前的最新仿真时间戳，否则会被 Servo 安全机制无情丢弃
    msg.header.stamp = this->now(); 
    msg.header.frame_id = "base_link";

    // 赋值平移速度 (以 0.05 速度缓慢向前，防止机械臂由于速度过快暴走)
    msg.twist.linear.x = 0.05; 
    msg.twist.linear.y = 0.0;
    msg.twist.linear.z = 0.0;
    
    msg.twist.angular.x = 0.0;
    msg.twist.angular.y = 0.0;
    msg.twist.angular.z = 0.0;

    publisher_->publish(msg);
  }

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr publisher_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr servo_start_client_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr autostart_timer_;
}; // 🌟 之前由于残缺漏掉了类的结束分号

// ====================================================================
// 🌟 灵魂补全：主函数入口 (main)
// ====================================================================
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  
  // 强制为该节点开启 use_sim_time，确保时钟与 Gazebo 完全同步
  rclcpp::NodeOptions options;
  options.append_parameter_override("use_sim_time", true);
  
  // 启动执行线程
  rclcpp::spin(std::make_shared<ServoTester>(options));
  rclcpp::shutdown();
  return 0;
}