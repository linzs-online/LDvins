cmake_minimum_required(VERSION 2.8.3)
project(vins)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

# 开启动态行人分割
set(open_segment_or_not no)
if(${open_segment_or_not} STREQUAL "Yes")
  message(=============================================================)
  message("Open Dynamic Obeject Segment function!")
  message(=============================================================)
  add_definitions(-DSegmentDynamic)
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.4")
  set(TENSORRT_DIR "/home/auto/lihao/download/softwares/TensorRT-8.4.2.4.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.2.4")
endif()

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models
    image_transport)

find_package(Ceres REQUIRED)
include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${Opencv_INCLUDE_DIRS})
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")



set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

if(${open_segment_or_not} STREQUAL "Yes")
include_directories(
  "/usr/local/include"
  ${TENSORRT_DIR}/samples/common
  ${TENSORRT_DIR}/include
  ${CUDA_TOOLKIT_ROOT_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include/
)

add_library(segement_lib src/dynamicSegment/dynamicSegment.cpp)
target_link_libraries(segement_lib ${OpenCV_LIBS}
                                   ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so
                                   ${TENSORRT_DIR}/lib/libnvinfer.so
                                   ${TENSORRT_DIR}/lib/libnvparsers.so
                                   ${TENSORRT_DIR}/lib/libnvinfer_plugin.so
                                   ${TENSORRT_DIR}/lib/libnvonnxparser.so)
endif()

catkin_package()

add_library(vins_lib
    src/estimator/parameters.cpp
    src/estimator/estimator.cpp
    src/estimator/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/featureTracker/feature_tracker.cpp)
target_link_libraries(vins_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})


add_executable(vins_node src/rosNodeTest.cpp)
target_link_libraries(vins_node vins_lib) 

if(${open_segment_or_not} STREQUAL "Yes")
  target_link_libraries(vins_node segement_lib
                                  ${TENSORRT_DIR}/targets/x86_64-linux-gnu/bin/common/logger.a)
endif()


# add_executable(kitti_odom_test src/KITTIOdomTest.cpp)
# target_link_libraries(kitti_odom_test vins_lib) 

# add_executable(kitti_gps_test src/KITTIGPSTest.cpp)
# target_link_libraries(kitti_gps_test vins_lib) 

