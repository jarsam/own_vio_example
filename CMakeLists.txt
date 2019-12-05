cmake_minimum_required(VERSION 2.8)
project(vio_example)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
find_package(GLEW REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
find_package(GSLAM REQUIRED)

include_directories(.
        ${PROJECT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${CERES_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
        ${GSLAM_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_library(MyVio
        System.cpp
        PinholeCamera.cpp
        FeatureTracker.cpp
        Estimator.cpp
)

target_link_libraries(MyVio
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        ${Pangolin_LIBRARIES}
        ${GLEW_LIBS}
        ${Boost_LIBRARIES}
        ${EIGEN3_LIBS}
        ${GSLAM_LIBS}
        )

add_executable(vio_example main.cpp)
target_link_libraries(vio_example MyVio)