#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <csignal>
#include <vector>
#include <mutex>

// =========================================================================
// ⭐ 终极控制节点：参数化速度、高精度扫描、安全平滑撤离
// =========================================================================

std::atomic<bool> g_shutdown_requested(false);
void sigint_handler(int signum) { (void)signum; g_shutdown_requested.store(true); }

class RobotControlSim : public rclcpp::Node
{
public:
    RobotControlSim() : Node("robot_control_sim"), has_trajectory_(false), is_task_completed_(false)
    {
        RCLCPP_INFO(this->get_logger(), "🤖 机械臂执行控制节点已启动！速度参数已加载。");

        // --- 声明可调节参数 ---
        // 1. 进场速度 (快速移动到起始点上方)
        this->declare_parameter("approach_vel", 0.3); // 默认 30% 速度
        this->declare_parameter("approach_acc", 0.2); 

        // 2. 扫描速度 (笛卡尔路径速度，建议设极低以实现 1mm/s)
        this->declare_parameter("scan_vel", 0.01);    // 极低速度模拟 1mm/s
        this->declare_parameter("scan_acc", 0.01);    

        // 3. 撤离速度 (返回 Home 姿态)
        this->declare_parameter("retract_vel", 0.2); 
        this->declare_parameter("retract_acc", 0.2);

        subscription_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/planned_poses", 10, std::bind(&RobotControlSim::trajectory_callback, this, std::placeholders::_1));

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500), std::bind(&RobotControlSim::control_loop, this));
    }

private:
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr subscription_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::vector<geometry_msgs::msg::Pose> target_poses_;
    std::mutex data_mutex_;
    bool has_trajectory_;
    bool is_task_completed_;
    std::string trajectory_frame_id_;

    void trajectory_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (has_trajectory_ || is_task_completed_) return; 

        target_poses_ = msg->poses;
        trajectory_frame_id_ = msg->header.frame_id; 
        has_trajectory_ = true;
        RCLCPP_INFO(this->get_logger(), "✅ 接收到新轨迹，准备执行。");
    }

    void control_loop()
    {
        if (g_shutdown_requested.load()) { rclcpp::shutdown(); return; }
        if (!has_trajectory_ || is_task_completed_) return;

        static const std::string PLANNING_GROUP = "ur7e_manipulater";
        auto move_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), PLANNING_GROUP);

        if (!trajectory_frame_id_.empty()) move_group->setPoseReferenceFrame(trajectory_frame_id_);

        // ---------------------------------------------------------
        // 1. 初始回 Home (使用进场速度)
        // ---------------------------------------------------------
        set_speed_params(move_group, "approach");
        RCLCPP_INFO(this->get_logger(), "🏠 正在前往初始 Home 姿态...");
        move_group->setNamedTarget("home_pose_1"); 
        move_group->move();

        // ---------------------------------------------------------
        // 2. 遍历执行各批次
        // ---------------------------------------------------------
        std::vector<geometry_msgs::msg::Pose> waypoints;
        const double JUMP_THRESHOLD = 0.05; 

        for (size_t i = 0; i < target_poses_.size(); ++i) {
            if (g_shutdown_requested.load()) break; 
            waypoints.push_back(target_poses_[i]);

            bool is_last_point = (i == target_poses_.size() - 1);
            bool is_jump = false;
            if (!is_last_point) {
                double dist = std::sqrt(std::pow(target_poses_[i].position.x - target_poses_[i+1].position.x, 2) +
                                        std::pow(target_poses_[i].position.y - target_poses_[i+1].position.y, 2) +
                                        std::pow(target_poses_[i].position.z - target_poses_[i+1].position.z, 2));
                if (dist > JUMP_THRESHOLD) is_jump = true;
            }

            if (is_jump || is_last_point) {
                execute_batch_optimized(move_group, waypoints);
                waypoints.clear(); 
            }
        }

        RCLCPP_INFO(this->get_logger(), "🎉 任务结束，机械臂已安全撤离。");
        is_task_completed_ = true; 
        has_trajectory_ = false; 
    }

    // ⭐ 核心逻辑：分阶段速度控制与安全撤离
    void execute_batch_optimized(std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group, 
                                 const std::vector<geometry_msgs::msg::Pose>& waypoints)
    {
        if (waypoints.empty() || g_shutdown_requested.load()) return;

        // --- 封装 Plan & Execute 动作 ---
        auto safe_move = [&](const geometry_msgs::msg::Pose& target, const std::string& name) {
            move_group->setPoseTarget(target);
            moveit::planning_interface::MoveGroupInterface::Plan plan;
            RCLCPP_INFO(this->get_logger(), "-> 执行: %s [X:%.3f, Y:%.3f, Z:%.3f]", name.c_str(), target.position.x, target.position.y, target.position.z);
            if (move_group->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
                move_group->execute(plan);
                return true;
            }
            return false;
        };

        // 阶段 A: 快速进场 (PTP 模式)
        set_speed_params(move_group, "approach");
        
        geometry_msgs::msg::Pose p1 = waypoints[0]; p1.position.z += 0.15;
        safe_move(p1, "1. 飞向高空过渡点");

        geometry_msgs::msg::Pose p2 = waypoints[0]; p2.position.z += 0.05;
        safe_move(p2, "2. 下降至进场点");

        safe_move(waypoints[0], "3. 接触起始点");

        // 阶段 B: 高精度扫描 (笛卡尔模式)
        set_speed_params(move_group, "scan");
        RCLCPP_INFO(this->get_logger(), "-> 4. 开始扫描 (1mm/s 匀速模式)...");
        
        moveit_msgs::msg::RobotTrajectory trajectory;
        // 允许接触工件表面 avoid_collisions = false
        double fraction = move_group->computeCartesianPath(waypoints, 0.005, 0.0, trajectory, false);

        if (fraction > 0.9) { 
            move_group->execute(trajectory);
            RCLCPP_INFO(this->get_logger(), "✔️ 扫描段执行成功。");
        } else {
            RCLCPP_ERROR(this->get_logger(), "❌ 扫描规划覆盖率过低 (%.2f%%)", fraction * 100.0);
        }

        // 阶段 C: 安全撤离 (改用返回 Home 姿态)
        set_speed_params(move_group, "retract");
        RCLCPP_INFO(this->get_logger(), "-> 5. 扫描结束，执行安全撤离动作...");
        
        // 改进点：不再做相对位移规划，直接规划回 Home 姿态以保证路径 100% 可达
        move_group->setNamedTarget("home_pose_1");
        moveit::planning_interface::MoveGroupInterface::Plan retract_plan;
        if (move_group->plan(retract_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            move_group->execute(retract_plan);
            RCLCPP_INFO(this->get_logger(), "✔️ 机械臂已安全返回 Home 姿态。");
        } else {
            RCLCPP_ERROR(this->get_logger(), "❌ 撤离规划失败！请检查是否有障碍物。");
        }
    }

    // 辅助函数：根据阶段切换速度
    void set_speed_params(std::shared_ptr<moveit::planning_interface::MoveGroupInterface> mg, const std::string& phase)
    {
        double v = 0.1, a = 0.1;
        if (phase == "approach") {
            v = this->get_parameter("approach_vel").as_double();
            a = this->get_parameter("approach_acc").as_double();
        } else if (phase == "scan") {
            v = this->get_parameter("scan_vel").as_double();
            a = this->get_parameter("scan_acc").as_double();
        } else if (phase == "retract") {
            v = this->get_parameter("retract_vel").as_double();
            a = this->get_parameter("retract_acc").as_double();
        }
        mg->setMaxVelocityScalingFactor(v);
        mg->setMaxAccelerationScalingFactor(a);
    }
};

int main(int argc, char** argv)
{
    std::signal(SIGINT, sigint_handler);
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RobotControlSim>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    while (rclcpp::ok() && !g_shutdown_requested.load()) {
        executor.spin_some(std::chrono::milliseconds(100));
    }
    rclcpp::shutdown();
    return 0;
}