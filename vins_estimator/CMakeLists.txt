cmake_minimum_required(VERSION 2.8.3)
project(vins)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

# 开启动态行人分割
set(open_segment_or_not yes)
if(${open_segment_or_not} STREQUAL "yes")
  message(=============================================================)
  message("Open Dynamic Obeject Segment function!")
  message(=============================================================)
  add_definitions(-DSegmentDynamic)
  set(CUDA_TOOLKIT_ROOT_DIR     "/usr/local/cuda-11.4")
  set(CUDNN_DIR    "/home/auto/libSrc/cudnn-8.6.0.163")
  set(TENSORRT_DIR "/home/auto/libSrc/TensorRT-8.4.3.1")
  set(PROTOBUF_DIR "/home/auto/libSrc/protobuf-3.6.1_install")

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
find_package(CUDA REQUIRED)
add_compile_options(-fPIC)
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/tensorRT
    ${PROJECT_SOURCE_DIR}/src/tensorRT/common
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${PROTOBUF_DIR}/include
    ${TENSORRT_DIR}/include
    ${CUDNN_DIR}/include
)
link_directories(
    ${PROTOBUF_DIR}/lib
    ${TENSORRT_DIR}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDNN_DIR}/lib
)

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

if(${open_segment_or_not} STREQUAL "yes")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -O0 -Wfatal-errors -pthread -w -g")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
  file(GLOB_RECURSE trtcpp_srcs ${PROJECT_SOURCE_DIR}/src/tensorRT/*.cpp)
  file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/src/*.cu)

  cuda_add_library(plugin_list SHARED ${cuda_srcs})
  target_link_libraries(plugin_list nvinfer nvinfer_plugin)
  target_link_libraries(plugin_list ${CUDA_LIBS} cublas cudart cudnn)
  target_link_libraries(plugin_list protobuf-lite pthread)
  target_link_libraries(plugin_list ${OpenCV_LIBS})

  add_library(trtPro ${trtcpp_srcs})
  target_link_libraries(trtPro nvinfer nvinfer_plugin)
  target_link_libraries(trtPro cuda cublas cudart cudnn)
  target_link_libraries(trtPro protobuf-lite pthread plugin_list)
  target_link_libraries(trtPro ${OpenCV_LIBS})
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

if(${open_segment_or_not} STREQUAL "yes")
  target_link_libraries(vins_node trtPro)
endif()


# add_executable(kitti_odom_test src/KITTIOdomTest.cpp)
# target_link_libraries(kitti_odom_test vins_lib) 

# add_executable(kitti_gps_test src/KITTIGPSTest.cpp)
# target_link_libraries(kitti_gps_test vins_lib) 

