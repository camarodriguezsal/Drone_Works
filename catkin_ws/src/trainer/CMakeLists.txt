cmake_minimum_required(VERSION 3.0.2)
project(trainer)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  geometry_msgs
  roscpp
  rospy
  std_msgs
)

add_service_files(
  FILES
  EnjoyStart.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs  
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/train.py
  scripts/enjoy.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
