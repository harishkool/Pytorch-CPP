cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


set(OpenCV_DIR /home/nagaharish/Downloads/OpenCV_installation/installation/OpenCV-master/lib/cmake/opencv4/)

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_executable(example-app src/example.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}" "{OpenCV_LIBRARY_DIRS}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)