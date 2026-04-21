cmake_minimum_required(VERSION 3.10)
project(CSRMatrix)

set(CMAKE_CXX_STANDARD 17)

add_executable(code main.cpp)
target_include_directories(code PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
