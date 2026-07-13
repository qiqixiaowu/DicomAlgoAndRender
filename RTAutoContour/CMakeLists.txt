cmake_minimum_required(VERSION 3.18)

# ── CUDA 可选支持 ──────────────────────────────────────────
option(USE_CUDA "Enable CUDA Marching Squares" OFF)

if(USE_CUDA)
    project(RTAutoContour LANGUAGES CXX CUDA)
else()
    project(RTAutoContour LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/MP>)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -DNOMINMAX)
endif()

# ── 收集源文件 ─────────────────────────────────────────────
file(GLOB RTAC_CPP_SOURCES CONFIGURE_DEPENDS src/*.cpp)
file(GLOB RTAC_HEADERS     CONFIGURE_DEPENDS include/*.hpp include/*.h)

set(RTAC_SOURCES ${RTAC_CPP_SOURCES})

if(USE_CUDA)
    file(GLOB RTAC_CU_SOURCES CONFIGURE_DEPENDS src/*.cu)
    list(APPEND RTAC_SOURCES ${RTAC_CU_SOURCES})
endif()

# ── 静态库 ─────────────────────────────────────────────────
add_library(rtac_algorithms STATIC ${RTAC_SOURCES} ${RTAC_HEADERS})

target_include_directories(rtac_algorithms PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(USE_CUDA)
    set_target_properties(rtac_algorithms PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION OFF
    )
    target_compile_definitions(rtac_algorithms PUBLIC USE_CUDA)
endif()

# ── VS 筛选器（源文件分组） ─────────────────────────────────
source_group("Header Files" FILES ${RTAC_HEADERS})
source_group("Source Files"  FILES ${RTAC_CPP_SOURCES})
if(USE_CUDA)
    source_group("CUDA Files" FILES ${RTAC_CU_SOURCES})
endif()

# ── 演示程序 ───────────────────────────────────────────────
add_executable(rtac_demo demo/main_demo.cpp)
target_link_libraries(rtac_demo PRIVATE rtac_algorithms)

if(MSVC)
    target_compile_options(rtac_demo PRIVATE /W3)
    # VS 中设置 rtac_demo 为启动项目
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT rtac_demo)
else()
    target_compile_options(rtac_demo PRIVATE -Wall -O2)
endif()

# ── 安装 ───────────────────────────────────────────────────
install(DIRECTORY include/ DESTINATION include/rtac)
install(TARGETS rtac_demo DESTINATION bin)