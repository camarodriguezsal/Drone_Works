cmake_minimum_required(VERSION 3.0.2)
project(unity_bridge)

find_package(catkin REQUIRED COMPONENTS
	geometry_msgs
	rospy
	std_msgs
	nav_msgs	
)

add_service_files(
  FILES
  SimulationMode.srv
  SimulationStep.srv
  SimulationReset.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs  
  nav_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
   scripts/unity_tcp_communication.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


include(ExternalProject)

ExternalProject_Add(unitysim
    URL               https://www.dropbox.com/s/0qc5mun35nbo6fz/RL_Drone_unity.zip?dl=0
    DOWNLOAD_NAME     RL_Drone_unity.zip
    #URL_HASH					MD5=e7de46d5197e182cf53b6f462b8956f4
    CONFIGURE_COMMAND chmod +x ../unitysim/RL_Drone.x86_64
    BUILD_COMMAND     ""
    INSTALL_COMMAND   cd ../unitysim/ && cp -rf . ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}
)

ExternalProject_Get_Property(unitysim DOWNLOAD_DIR)
set_property(DIRECTORY PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    "${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/."
)

message(STATUS "UNITYSIM DOWNLOAD_DIR: ${DOWNLOAD_DIR}")


