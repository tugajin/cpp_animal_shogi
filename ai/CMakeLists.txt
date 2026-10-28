cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

# Project Name
project(cpp_animal_shogi CXX)

# ONNX Runtime 1.20.0 (GPU build), placed next to ~/libtorch
if(NOT DEFINED ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime-1.20.0")
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

# Find Package
find_package(Threads REQUIRED)

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g -DNDEBUG -march=native -fsanitize=thread")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -s -DNDEBUG -march=native")

# Create Executable File
add_executable(${PROJECT_NAME} main.cpp)
target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Wextra)
target_link_libraries(${PROJECT_NAME} onnxruntime)
target_link_libraries(${PROJECT_NAME} "-pthread" "-fsanitize=thread" "-B/usr/lib/gcc/x86_64-linux-gnu/13")

# Find libonnxruntime.so at runtime without LD_LIBRARY_PATH
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)