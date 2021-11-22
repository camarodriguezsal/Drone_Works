cmake_minimum_required(VERSION 3.0.2)
project(error_viz)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  tf
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
   scripts/visualizer.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


