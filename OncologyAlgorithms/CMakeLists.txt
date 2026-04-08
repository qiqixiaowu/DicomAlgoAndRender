cmake_minimum_required(VERSION 3.18)
project(OncologyAlgorithms VERSION 1.0.0 LANGUAGES CXX)

# ─────────────────────────────────────────────
#  C++ 标准
# ─────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─────────────────────────────────────────────
#  可选：CUDA 支持（深度学习 nnUNet 推理）
# ─────────────────────────────────────────────
option(USE_CUDA "Enable CUDA/cuDNN GPU inference for DL lung nodule module" OFF)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_library(CUDNN_LIB cudnn
        HINTS "${CUDAToolkit_LIBRARY_DIR}"
              "$ENV{CUDNN_PATH}/lib/x64"
              "C:/Program Files/NVIDIA/CUDNN/v8.x/lib/x64")
    if(NOT CUDNN_LIB)
        message(FATAL_ERROR "cuDNN not found. Set CUDNN_PATH or disable USE_CUDA.")
    endif()
    message(STATUS "CUDA enabled. cuDNN: ${CUDNN_LIB}")
endif()

# ─────────────────────────────────────────────
#  Include 目录
# ─────────────────────────────────────────────
set(INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# ─────────────────────────────────────────────
#  OncologyAlgorithms 静态库
#
#  不依赖任何第三方库，可在 CPU-only 环境直接编译
#  CUDA 路径通过 USE_CUDA 宏控制
# ─────────────────────────────────────────────
file(GLOB ONC_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
if(USE_CUDA)
    file(GLOB ONC_CUDA_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
    list(APPEND ONC_SRCS ${ONC_CUDA_SRCS})
endif()

add_library(OncologyAlgorithms STATIC ${ONC_SRCS})
target_include_directories(OncologyAlgorithms PUBLIC ${INC_DIR})

if(USE_CUDA)
    target_compile_definitions(OncologyAlgorithms PUBLIC USE_CUDA)
    set_target_properties(OncologyAlgorithms PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;86;89")    # Turing, Ampere, Ada Lovelace
    target_link_libraries(OncologyAlgorithms PUBLIC
        CUDA::cudart
        ${CUDNN_LIB})
    target_include_directories(OncologyAlgorithms PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS})
endif()

# ─────────────────────────────────────────────
#  编译选项
# ─────────────────────────────────────────────
if(MSVC)
    target_compile_options(OncologyAlgorithms PRIVATE
        /W3 /MP          # 多处理器编译
        /fp:fast         # 快速浮点
        /openmp          # OpenMP（加速 CPU 推理循环）
        /utf-8)          # 将源文件和执行字符集均设为 UTF-8（防止中文注释乱码）
    target_compile_definitions(OncologyAlgorithms PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX)
else()
    target_compile_options(OncologyAlgorithms PRIVATE
        -Wall -O3 -march=native -fopenmp -ffast-math
        -finput-charset=UTF-8 -fexec-charset=UTF-8)
endif()

# ─────────────────────────────────────────────
#  演示可执行文件
# ─────────────────────────────────────────────
add_executable(OncDemo "${CMAKE_CURRENT_SOURCE_DIR}/demo/main.cpp")
target_link_libraries(OncDemo PRIVATE OncologyAlgorithms)
target_include_directories(OncDemo PRIVATE ${INC_DIR})
if(MSVC)
    target_compile_options(OncDemo PRIVATE /W3 /MP /utf-8)
endif()

# ─────────────────────────────────────────────
#  安装规则（可选）
# ─────────────────────────────────────────────
install(TARGETS OncologyAlgorithms
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/
        DESTINATION include)

# ─────────────────────────────────────────────
#  打印配置摘要
# ─────────────────────────────────────────────
message(STATUS "=== OncologyAlgorithms Configuration ===")
message(STATUS "  CUDA support : ${USE_CUDA}")
message(STATUS "  C++ standard : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
