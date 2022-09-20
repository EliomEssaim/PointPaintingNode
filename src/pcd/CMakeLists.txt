cmake_minimum_required(VERSION 2.8.3)
project(lidar_msgs)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS 
  message_generation
  std_msgs
  geometry_msgs
  sensor_msgs
)
# 需要加入 message_generation,必须有 std_msgs


#######################################
## Declare ROS messages and services ##
#######################################

## Generate messages in the 'msg' folder
## 配置 msg 源文件
add_message_files(DIRECTORY msg
  FILES
  Lane.msg
  LaneArray.msg
  Waypoint.msg
  WaypointState.msg
  DTLane.msg
  DetectedObject.msg
  DetectedObjectArray.msg
)

## Generate services in the 'srv' folder
add_service_files(
	DIRECTORY srv
	FILES
)

## Generate added messages and services with any dependencies listed here
# 生成消息时依赖于 std_msgs
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

#执行时依赖
catkin_package(
  CATKIN_DEPENDS
  std_msgs
  geometry_msgs
  sensor_msgs
  message_runtime
)
