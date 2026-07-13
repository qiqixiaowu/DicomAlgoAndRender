cmake_minimum_required(VERSION 3.14)
project(MedicalSegmentation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC UTF-8 源文件支持
if(MSVC)
    add_compile_options(/utf-8)
endif()

# Header-only 算法库（所有实现在 .hpp 中）
add_library(medseg_algorithms INTERFACE)
target_include_directories(medseg_algorithms INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 演示可执行程序
add_executable(medseg_demo demo/main_demo.cpp)
target_link_libraries(medseg_demo PRIVATE medseg_algorithms)

# 安装
install(DIRECTORY include/ DESTINATION include/medseg)
install(TARGETS medseg_demo DESTINATION bin)
