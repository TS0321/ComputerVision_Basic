SET(execName ArUcoPoseEstimation)
cmake_minimum_required(VERSION 2.8)
PROJECT(${execName})

FILE(GLOB SRC_FILES *.cpp)
FILE(GLOB HDR_FILES *.h)

set(YAML_CPP_INCLUDE_DIR "C:/lib/yaml-cpp-master/build/install/include")
set(YAML_CPP_LIBRARIES "C:/lib/yaml-cpp-master/build/install/lib")

link_directories(${YAML_CPP_LIBRARIES})
add_executable(${execName} ${HDR_FILES} ${SRC_FILES})
find_package(OpenCV REQUIRED)
target_link_libraries(${execName} ${OpenCV_LIBS})
target_include_directories(${execName} PUBLIC ${OpenCV_INCLUDE_DIRS})
#find_package(yaml-cpp REQUIRED)
#include_directories(${YAML_CPP_INCLUDE_DIR})
#target_include_directories(${execName} PUBLIC ${YAML_CPP_INCLUDE_DIR})
#target_link_libraries(${execName} ${YAML_CPP_LIBRARIES})
target_include_directories(${execName} PUBLIC ${YAML_CPP_INCLUDE_DIR})
target_link_libraries(${execName} PUBLIC yaml-cpp.lib)
find_package(Eigen3 REQUIRED)
target_include_directories(${execName} PUBLIC ${EIGEN3_INCLUDE_DIRS})


