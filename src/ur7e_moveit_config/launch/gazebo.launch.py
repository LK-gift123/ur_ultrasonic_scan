import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, Command
from launch.event_handlers import OnProcessExit
from ament_index_python.packages import get_package_share_directory

import xacro
import re

def remove_comments(text):
    pattern = r'<!--(.*?)-->'
    return re.sub(pattern, '', text, flags=re.DOTALL)

def generate_launch_description():
    robot_name_in_model = 'ur7e'
    package_name = 'mybot_description'
    urdf_name = "ur7e.urdf"

    pkg_share = FindPackageShare(package=package_name).find(package_name) 
    urdf_model_path = os.path.join(pkg_share, f'urdf/{urdf_name}')
    table_xacro_path = os.path.join(pkg_share, 'urdf/table.xacro')
    dummy_moveit_config = FindPackageShare(package='ur7e_moveit_config').find('ur7e_moveit_config')
    gazebo_world_path = os.path.join(pkg_share, 'world/myworld.world')  # 如有需要可启用

        # 定义一个临时 URDF 文件路径，用于存放 xacro 处理后的结果
    table_urdf_path = '/tmp/table.urdf'

    # 1. 执行 xacro 命令，将 table.xacro 转换为 URDF 文件
    generate_table_urdf = ExecuteProcess(
        cmd=['xacro', table_xacro_path, '-o', table_urdf_path],
        output='screen',
        name='generate_table_urdf'   # 给动作起个名字，方便调试
    )

    # 2. 使用 spawn_entity.py 将生成的 URDF 文件加载到 Gazebo 中
    spawn_table_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'workpieces', '-file', table_urdf_path,'-x', '0.4', '-y', '0.0', '-z', '0.05'],
        output='screen'
    )

    # 3. 为了确保 xacro 处理完成后再 spawn，可以用 RegisterEventHandler 连接两者
    #    但 spawn_entity.py 内部会等待 Gazebo 就绪，因此直接顺序添加也可以。
    #    这里为了保险，采用事件处理：generate_table_urdf 退出后执行 spawn_table_cmd
    table_spawn_after_generate = RegisterEventHandler(
        OnProcessExit(
            target_action=generate_table_urdf,
            on_exit=[spawn_table_cmd]
        )
    )

    controller_config = PathJoinSubstitution(
        [dummy_moveit_config, 'config', 'ros2_controllers.yaml']
    )

    # 启动 Gazebo 服务器
    start_gazebo_cmd = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )
    # 如果需要加载世界文件，取消下面行的注释并注释掉上一行
    # start_gazebo_cmd = ExecuteProcess(
    #     cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so', gazebo_world_path],
    #     output='screen'
    # )

    # 处理 Xacro 文件生成 robot_description
    doc = xacro.parse(open(urdf_model_path))
    xacro.process_doc(doc)
    robot_description_content = remove_comments(doc.toxml())
    robot_description = {'robot_description': robot_description_content}

    # robot_state_publisher 节点
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'use_sim_time': True}, robot_description, {"publish_frequency": 15.0}],
        output='screen'
    )

    # 在 Gazebo 中生成机器人实体（依赖 robot_description 话题）
    spawn_entity_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', robot_name_in_model, '-topic', 'robot_description'],
        output='screen'
    )

    # ========== 使用 spawner 加载控制器（替代 ExecuteProcess） ==========
    # joint_state_broadcaster 加载器
    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    # 轨迹控制器加载器（名称需与 ros2_controllers.yaml 一致）
    load_joint_trajectory_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['ur7e_manipulater_controller', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    # 控制启动顺序：在机器人生成后加载 joint_state_broadcaster
    close_evt1 = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_entity_cmd,
            on_exit=[load_joint_state_broadcaster],
        )
    )
    # 在 joint_state_broadcaster 加载成功后加载轨迹控制器
    close_evt2 = RegisterEventHandler(
        OnProcessExit(
            target_action=load_joint_state_broadcaster,
            on_exit=[load_joint_trajectory_controller],
        )
    )

    ld = LaunchDescription()
    ld.add_action(start_gazebo_cmd)
    ld.add_action(robot_state_publisher)
    # [新增] 先生成桌子
    ld.add_action(generate_table_urdf)
    ld.add_action(table_spawn_after_generate)   # 这个事件会触发 spawn_table_cmd
    ld.add_action(spawn_entity_cmd)
    ld.add_action(close_evt1)
    ld.add_action(close_evt2)

    return ld