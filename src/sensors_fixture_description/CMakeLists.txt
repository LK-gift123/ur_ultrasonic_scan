cmake_minimum_required(VERSION 3.8)
project(sensors_fixture_description)

# 查找 ament_cmake
find_package(ament_cmake REQUIRED)

# 安装目录（urdf, meshes, config, launch 等）
install(DIRECTORY config launch meshes urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()