cmake_minimum_required(VERSION 3.8)
project(yolo_ballon_inference_pipeline)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BoundingBox.msg"
  "msg/BoundingBoxes.msg"
  DEPENDENCIES std_msgs
)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install Python scripts
install(PROGRAMS
yolo_ballon_inference_pipeline/yolo_ballon_inference_pipeline.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
