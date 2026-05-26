import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # 使用 MoveItConfigsBuilder 构建完整配置字典（包含 URDF, SRDF, kinematics, joint_limits）
    moveit_config = (
        MoveItConfigsBuilder("ur7e", package_name="ur7e_moveit_config")
        .to_moveit_configs()
    )
    moveit_dict = moveit_config.to_dict()

    # 加载伺服参数 YAML 文件（可选）
    servo_yaml = os.path.join(
        get_package_share_directory("ur7e_moveit_config"),
        "config",
        "servo_params.yaml"
    )

    # 核心伺服参数（确保关键字段正确）
    servo_params = {
        "moveit_servo": {
            # 输入：使用速度单位（m/s, rad/s）
            "command_in_type": "speed_units",
            "cartesian_command_in_topic": "/servo_node/delta_twist_cmds",
            "joint_command_in_topic": "/servo_node/delta_joint_cmds",

            # 输出：发布到速度控制器
            "command_out_topic": "/forward_velocity_controller/commands",
            "command_out_type": "std_msgs/Float64MultiArray",

            # 发布速度，不发布位置
            "publish_joint_positions": False,
            "publish_joint_velocities": True,
            "publish_joint_accelerations": False,

            # 机器人参数（必须与 SRDF/URDF 一致）
            "move_group_name": "ur7e_manipulator",
            "planning_frame": "base_link",
            "ee_frame_name": "ft_frame",      # 请确认 URDF 中存在该 link

            # 控制周期与调试
            "publish_period": 0.02,
            "low_latency_mode": False,
            "log_level": "debug",             # 输出详细日志

            # 安全限制
            "incoming_command_timeout": 0.1,
            "num_outgoing_halt_msgs_to_publish": 4,
            "lower_singularity_threshold": 17.0,
            "hard_stop_singularity_threshold": 30.0,
            "joint_limit_margin": 0.1,
            "check_collisions": False,
        }
    }

    # 创建 servo_node 并传递完整配置
    servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="servo_node",
        output="screen",
        parameters=[
            moveit_dict,           # 包含 URDF, SRDF, kinematics, joint_limits
            servo_params,          # 伺服专用参数
            {"use_sim_time": True},# 仿真时间
            servo_yaml,            # 外部 YAML（可选，用作后备）
        ]
    )

    return LaunchDescription([servo_node])