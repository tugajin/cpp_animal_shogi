cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project Name
project(cpp_animal_shogi CXX)

set(CMAKE_PREFIX_PATH "~/libtorch")
set(CMAKE_BUILD_TYPE Debug)

# Find Package
find_package(Torch REQUIRED)


# Create Executable File
add_executable(${PROJECT_NAME} main.cpp)
target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Wextra)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_compile_definitions(${PROJECT_NAME}  PRIVATE DEBUG=1)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)