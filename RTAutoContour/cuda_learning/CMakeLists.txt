# ==============================================================
#  CUDA 学习系列 - CMakeLists.txt
# ==============================================================
#  编译方法 (使用 junction 路径避免中文路径问题):
#    cd C:\build\RTAutoContour
#    cmake -S cuda_learning -B cuda_learning/build -G "Visual Studio 17 2022" -DCMAKE_CUDA_ARCHITECTURES=86
#    cmake --build cuda_learning/build --config Release
#
#  或者逐个编译:
#    nvcc 01_hello_cuda.cu -o 01_hello_cuda.exe
#    nvcc 02_vector_add.cu -o 02_vector_add.exe
#    ...
# ==============================================================

cmake_minimum_required(VERSION 3.18)
project(CudaLearning LANGUAGES CXX CUDA)

# CUDA 架构 (根据你的 GPU 修改)
# RTX 30xx → 86, RTX 40xx → 89, RTX 20xx → 75
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# MSVC 编码设置 (只对 CXX，不传给 nvcc)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)

# 每个 .cu 文件生成一个独立的可执行文件
set(LESSONS
    01_hello_cuda
    02_vector_add
    03_matrix_mul
    04_reduction
    05_histogram
    06_streams
)

foreach(lesson ${LESSONS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${lesson}.cu")
        add_executable(${lesson} ${lesson}.cu)
        set_target_properties(${lesson} PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
        # VS 中的文件夹组织
        set_target_properties(${lesson} PROPERTIES FOLDER "CUDA Learning")
    endif()
endforeach()

message(STATUS "CUDA Learning: ${LESSONS}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
