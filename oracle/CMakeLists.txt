cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project Name
project(doubutsu CXX)

set(CMAKE_BUILD_TYPE Debug)

# Find Package

set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg -DDEBUG")

# Create Executable File
add_executable(${PROJECT_NAME} main.cpp allStateTable.cc dobutsu.cc winLoseTable.cc)
target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Wextra)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
