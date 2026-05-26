import os
import re
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit
from ament_index_python.packages import get_package_share_directory
import xacro
from launch.actions import TimerAction

# ====================================================================
# ⭐ 终极清洗函数：拦截器函数 + 清除换行符
# 作用：将 package:// 替换为物理硬盘绝对路径 file://，并压缩成单行，绕过解析 Bug
# ====================================================================
def clean_urdf(text):
    # 1. 移除 XML 声明
    text = re.sub(r'<\?xml.*?\?>', '', text)
    # 2. 移除所有注释
    text = re.sub(r'', '', text, flags=re.DOTALL)
    
    # 3. 将 package:// 替换为物理硬盘绝对路径
    def resolve_package_path(match):
        pkg_name = match.group(1)
        rel_path = match.group(2)
        try:
            pkg_path = get_package_share_directory(pkg_name)
            return f"file://{pkg_path}/{rel_path}"
        except Exception:
            return match.group(0)
            
    text = re.sub(r'package://([a-zA-Z0-9_]+)/([a-zA-Z0-9_./-]+)', resolve_package_path, text)
    
    # 4. 🌟最关键一步：移除所有的换行符和回车符，压成单行！
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def generate_launch_description():
    robot_name_in_model = 'ur7e'
    package_name = 'mybot_description'
    urdf_name = "ur7e.urdf.xacro"

    pkg_share = FindPackageShare(package=package_name).find(package_name) 
    urdf_model_path = os.path.join(pkg_share, f'urdf/{urdf_name}')
    table_xacro_path = os.path.join(pkg_share, 'urdf/table.xacro')
    table_urdf_path = '/tmp/table.urdf'

    generate_table_urdf = ExecuteProcess(
        cmd=['xacro', table_xacro_path, '-o', table_urdf_path],
        output='screen',
        name='generate_table_urdf'
    )

    # ⚠️ 修复点 1：坐标必须是算好的结果，不能写算式 '-0.325+0.3'
    spawn_table_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'workpieces', '-file', table_urdf_path, '-x', '-0.025', '-y', '-0.625', '-z', '0.8'],
        output='screen'
    )

    table_spawn_after_generate = RegisterEventHandler(
        OnProcessExit(
            target_action=generate_table_urdf,
            on_exit=[spawn_table_cmd]
        )
    )

    # ====================================================================
    # ⚠️ 修复点 2：处理异形曲面的路径和 Mesh 加载问题
    # ====================================================================
    workpiece_urdf_path = os.path.join(
        get_package_share_directory('artifacts_workpiece_description'), 
        'urdf', 
        'artifacts_workpiece_description.urdf'
    )
    
    # 读取异形曲面的 URDF，并进行 package:// 路径清洗，防止 Gazebo 找不到网格
    with open(workpiece_urdf_path, 'r') as f:
        workpiece_content = f.read()
        processed_workpiece_content = clean_urdf(workpiece_content)
    
    # 存入临时文件供 Gazebo 读取
    processed_workpiece_path = '/tmp/processed_custom_surface.urdf'
    with open(processed_workpiece_path, 'w') as f:
        f.write(processed_workpiece_content)

    # 生成异形曲面 (由于桌子表面是0.9，曲面初始化也放 0.9 或者 0.95 避免穿模)
    spawn_custom_workpiece_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'custom_surface', '-file', processed_workpiece_path, '-x', '0.025', '-y', '-0.625', '-z', '0.8'],
        output='screen'
    )

    start_gazebo_cmd = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    # 解析主机械臂 Xacro 并调用清洗函数
    doc = xacro.parse(open(urdf_model_path))
    xacro.process_doc(doc)
    
    # ⚠️ 修复点 3：这里必须调用 clean_urdf，压缩换行符
    robot_description_content = clean_urdf(doc.toxml())
    robot_description = {'robot_description': robot_description_content}

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'use_sim_time': True}, robot_description, {"publish_frequency": 15.0}],
        output='screen'
    )

    spawn_entity_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', robot_name_in_model, '-topic', 'robot_description'],
        output='screen'
    )

    load_controllers = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            'ur7e_manipulator_controller',
            'forward_velocity_controller',
            '--controller-manager',
            '/controller_manager'
        ],
        output='screen'
    )

    delay_controller_loading = TimerAction(
        period=3.0,
        actions=[load_controllers]
    )

    ld = LaunchDescription()
    ld.add_action(start_gazebo_cmd)
    ld.add_action(robot_state_publisher)
    ld.add_action(generate_table_urdf)
    # ld.add_action(table_spawn_after_generate)
    ld.add_action(spawn_custom_workpiece_cmd)  # 加入异形曲面节点
    ld.add_action(spawn_entity_cmd)
    ld.add_action(delay_controller_loading)

    return ld