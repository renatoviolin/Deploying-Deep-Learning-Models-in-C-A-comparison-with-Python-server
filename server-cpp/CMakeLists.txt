set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include_directories("/opt/X11/include")
include_directories("/usr/local/include/")
include_directories("/usr/local/include/opencv4")

set(Torch_DIR "/Users/renato/Documents/dev/libtorch/share/cmake/Torch")
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(server-cpp)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(server-cpp server.cpp libs/infer.cpp libs/base64.cpp libs/opencvutils.cpp libs/torchutils.cpp)

target_link_libraries(server-cpp "${TORCH_LIBRARIES}")
target_link_libraries(server-cpp "${OpenCV_LIBS}")
target_link_libraries(server-cpp ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

set_property(TARGET server-cpp PROPERTY CXX_STANDARD 14)
set_property(TARGET server-cpp PROPERTY OUTPUT_NAME server-app)